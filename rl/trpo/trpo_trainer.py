import torch
import torch.nn.functional as F
import numpy as np
import math
from config import Config


class TRPO:

    def __init__(self, model):

        self.model = model

        self.gamma = Config.TRPO_GAMMA
        self.lam = Config.TRPO_LAMBDA

        self.max_kl = Config.TRPO_MAX_KL
        self.damping = Config.TRPO_DAMPING

        self.cg_iters = Config.TRPO_CG_ITERS

        self.entropy_coef = 0.005   # ⭐ 新增（防坍塌）

        self.value_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.VALUE_LR
        )

    # =========================
    # Returns（修复）
    # =========================
    def compute_returns(self, rewards, dones):

        returns = []
        R = 0

        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)

        return torch.tensor(returns, dtype=torch.float32)

    # =========================
    # GAE
    # =========================
    def compute_gae(self, rewards, values, dones):

        adv = []
        gae = 0

        values = values + [0]

        for i in reversed(range(len(rewards))):

            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]

            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            adv.insert(0, gae)

        adv = torch.tensor(adv, dtype=torch.float32)

        # ⭐ 标准化 + 裁剪（防爆）
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = torch.clamp(adv, -5, 5)

        return adv

    # =========================
    # KL
    # =========================
    def get_kl(self, states, old_probs):

        logits, _ = self.model(states)
        new_probs = F.softmax(logits, dim=-1)

        kl = (old_probs * (torch.log(old_probs + 1e-8) - torch.log(new_probs + 1e-8))).sum(dim=1)

        return kl.mean()

    # =========================
    # loss（加入entropy）
    # =========================
    def get_loss(self, states, actions, old_probs, advs):

        logits, _ = self.model(states)

        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, 1e-8, 1.0)  # ⭐ 防NaN

        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)

        old_log_probs = torch.log(
            old_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8
        )

        ratio = torch.exp(log_probs - old_log_probs)

        surrogate = ratio * advs

        # ⭐ entropy
        entropy = dist.entropy().mean()

        loss = -(surrogate.mean() + self.entropy_coef * entropy)

        return loss

    # =========================
    # FVP
    # =========================
    def fisher_vector_product(self, states, old_probs, v):

        kl = self.get_kl(states, old_probs)

        grads = torch.autograd.grad(
            kl,
            self.model.parameters(),
            create_graph=True,
            allow_unused=True
        )

        grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, self.model.parameters())
        ]

        flat_grad_kl = torch.cat([g.view(-1) for g in grads])

        kl_v = (flat_grad_kl * v).sum()

        grads = torch.autograd.grad(
            kl_v,
            self.model.parameters(),
            allow_unused=True
        )

        grads = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(grads, self.model.parameters())
        ]

        flat_grad_grad_kl = torch.cat([g.contiguous().view(-1) for g in grads]).detach()

        return flat_grad_grad_kl + self.damping * v

    # =========================
    # CG
    # =========================
    def conjugate_gradients(self, Avp, b):

        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()

        rdotr = torch.dot(r, r)

        for _ in range(self.cg_iters):

            Avp_p = Avp(p)
            alpha = rdotr / (torch.dot(p, Avp_p) + 1e-8)

            x += alpha * p
            r -= alpha * Avp_p

            new_rdotr = torch.dot(r, r)

            if new_rdotr < 1e-10:
                break

            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    # =========================
    # 参数处理
    # =========================
    def get_flat_params(self):
        return torch.cat([p.data.view(-1) for p in self.model.parameters()])

    def set_flat_params(self, flat):

        idx = 0
        for p in self.model.parameters():
            size = p.numel()
            p.data = flat[idx:idx+size].view(p.size())
            idx += size

    # =========================
    # 主更新（增强版）
    # =========================
    def update(self, states, actions, old_probs, rewards, dones, global_steps):
        self.entropy_coef = 0.001 + 0.009 / (1 + math.exp(0.0001 * (global_steps - 50000)))
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions)
        old_probs = torch.tensor(np.array(old_probs), dtype=torch.float32)

        # -------- value --------
        _, values = self.model(states)
        values = values.squeeze().detach().tolist()

        advs = self.compute_gae(rewards, values, dones)
        returns = self.compute_returns(rewards, dones)

        # -------- loss grad --------
        loss = self.get_loss(states, actions, old_probs, advs)

        grads = torch.autograd.grad(loss, self.model.parameters(), allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, self.model.parameters())]

        loss_grad = torch.cat([g.view(-1) for g in grads]).detach()

        # -------- FVP --------
        def Avp(v):
            return self.fisher_vector_product(states, old_probs, v)

        step_dir = self.conjugate_gradients(Avp, -loss_grad)

        shs = 0.5 * (step_dir * Avp(step_dir)).sum()
        step = step_dir / (torch.sqrt(shs / self.max_kl) + 1e-8)

        old_params = self.get_flat_params()
        old_loss = loss.detach()

        # ⭐ line search（KL + Loss 双约束）
        success = False

        for step_frac in [1.0, 0.5, 0.25, 0.125, 0.0625]:

            new_params = old_params + step_frac * step
            self.set_flat_params(new_params)

            new_loss = self.get_loss(states, actions, old_probs, advs)
            kl = self.get_kl(states, old_probs)

            if kl < self.max_kl and new_loss < old_loss:
                success = True
                break

        if not success:
            self.set_flat_params(old_params)

        # -------- value update --------
        for _ in range(5):
            _, v = self.model(states)
            value_loss = F.mse_loss(v.squeeze(), returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        return kl.item()