import numpy as np
from rl.trpo.cg_solver import conjugate_gradient


class TRPOAgent:

    def __init__(self, net):
        self.net = net

        self.gamma = 0.99
        self.lam = 0.95

        self.delta = 0.005
        self.entropy_coef = 0.001
        self.value_lr = 1e-3

    # =========================
    # 采样
    # =========================
    def sample_action(self, state):
        probs, _, _, _ = self.net.forward(state)
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    # =========================
    # GAE
    # =========================
    def compute_gae(self, rewards, values, dones):

        adv = []
        gae = 0

        for i in reversed(range(len(rewards))):

            next_v = values[i + 1] if i < len(rewards) - 1 else 0
            mask = 1.0 - dones[i]

            delta = rewards[i] + self.gamma * next_v * mask - values[i]

            gae = delta + self.gamma * self.lam * mask * gae
            adv.insert(0, gae)

        adv = np.array(adv)

        # ⭐ 只标准化
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return adv

    # =========================
    # Value更新（TD）
    # =========================
    def update_value(self, states, rewards, dones):

        for i in range(len(states)):

            s = states[i]

            _, _, v, h_v = self.net.forward(s)

            if i < len(states) - 1:
                next_v = self.net.forward(states[i + 1])[2]
                done = dones[i]
            else:
                next_v = 0
                done = 1

            target = rewards[i] + self.gamma * next_v * (1 - done)

            td = target - v
            td = np.clip(td, -1, 1)

            dw2 = np.outer(h_v, [td])
            db2 = np.array([td])

            dh = self.net.w2_v.flatten() * td
            dh = (1 - h_v ** 2) * dh

            dw1 = np.outer(s, dh)
            db1 = dh

            self.net.w2_v += self.value_lr * dw2
            self.net.b2_v += self.value_lr * db2
            self.net.w1_v += self.value_lr * dw1
            self.net.b1_v += self.value_lr * db1

    # =========================
    # Policy Gradient
    # =========================
    def compute_policy_grad(self, states, actions, old_probs, advs):

        grads = []

        for s, a, adv in zip(states, actions, advs):

            probs, h, _, _ = self.net.forward(s)

            dlog = -probs
            dlog[a] += 1

            entropy_grad = -(np.log(probs + 1e-8) + 1)

            dlog = dlog * adv + self.entropy_coef * entropy_grad

            dw2 = np.outer(h, dlog)
            db2 = dlog

            dh = np.dot(self.net.w2, dlog)
            dh = (1 - h ** 2) * dh

            dw1 = np.outer(s, dh)
            db1 = dh

            grad = np.concatenate([
                dw1.flatten(),
                db1.flatten(),
                dw2.flatten(),
                db2.flatten()
            ])

            grads.append(grad)

        g = np.mean(grads, axis=0)

        if np.isnan(g).any():
            print("[WARN] grad NaN")
            return None

        return g

    # =========================
    # KL
    # =========================
    def mean_kl(self, states, old_probs):

        kl = 0

        for s, old_p in zip(states, old_probs):

            new_p, _, _, _ = self.net.forward(s)

            kl += np.sum(old_p * (np.log(old_p + 1e-8) - np.log(new_p + 1e-8)))

        return kl / len(states)

    # =========================
    # Fisher
    # =========================
    def fisher_vector_product(self, states, old_probs, v):

        damping = 1e-2
        fisher_v = np.zeros_like(v)

        for s, old_p in zip(states, old_probs):

            probs, h, _, _ = self.net.forward(s)

            for a in range(len(probs)):

                # ===== 1️⃣ 计算 ∇ log π(a|s) =====
                dlog = -probs
                dlog[a] += 1  # one-hot trick

                # ===== 反向传播 =====
                dw2 = np.outer(h, dlog)
                db2 = dlog

                dh = np.dot(self.net.w2, dlog)
                dh = (1 - h ** 2) * dh

                dw1 = np.outer(s, dh)
                db1 = dh

                g = np.concatenate([
                    dw1.flatten(),
                    db1.flatten(),
                    dw2.flatten(),
                    db2.flatten()
                ])

                # ===== 2️⃣ Fisher 累加 =====
                fisher_v += probs[a] * np.dot(g, v) * g

        fisher_v /= len(states)

        return fisher_v + damping * v * v


    def surrogate_loss(self, states, actions, old_probs, advs):

        loss = 0

        for s, a, old_p, adv in zip(states, actions, old_probs, advs):
            new_p, _, _, _ = self.net.forward(s)
            ratio = new_p[a] / (old_p[a] + 1e-8)
            loss += ratio * adv

        return loss / len(states)


    # =========================
    # Line Search
    # =========================
    def line_search(self, states, actions, old_probs, advs, old_params, step):

        old_loss = self.surrogate_loss(states, actions, old_probs, advs)

        step_frac = 1.0

        for _ in range(10):

            new_params = old_params + step_frac * step
            self.net.set_params(new_params)

            kl = self.mean_kl(states, old_probs)
            new_loss = self.surrogate_loss(states, actions, old_probs, advs)

            if kl < self.delta and new_loss > old_loss:
                return new_params

            step_frac *= 0.5

        return old_params

    # =========================
    # 主更新
    # =========================
    def update(self, states, actions, old_probs, rewards, dones):

        values = [self.net.forward(s)[2] for s in states]

        advs = self.compute_gae(rewards, values, dones)

        g = self.compute_policy_grad(states, actions, old_probs, advs)

        if g is None:
            return

        def Avp(v):
            return self.fisher_vector_product(states, old_probs, v)

        step_dir = conjugate_gradient(Avp, g)

        shs = 0.5 * np.dot(step_dir, Avp(step_dir))

        if shs <= 0:
            print("[WARN] bad curvature, using fallback")
            step_dir = g
            shs = np.dot(g, g)

        step = step_dir / (np.sqrt(shs / self.delta) + 1e-8)

        old_params = self.net.get_params()

        new_params = self.line_search(
            states, actions, old_probs, advs, old_params, step
        )
        self.net.set_params(new_params)

        # ⭐ 更新 value
        self.update_value(states, rewards, dones)
        
        return self.mean_kl(states, old_probs)