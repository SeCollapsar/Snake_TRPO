import torch
import numpy as np
import os

from env.snake_env import SnakeEnv
from rl.trpo.trpo_model import ActorCritic
from rl.trpo.trpo_trainer import TRPO
from config import Config
from utils.model_manager import ModelManager


env = SnakeEnv()

state_dim = Config.GRID_SIZE * Config.GRID_SIZE
action_dim = Config.ACTIONS

model = ActorCritic(state_dim, action_dim)
global_steps = 0

if os.path.exists(Config.LATEST_MODEL):

    # model.load_state_dict(torch.load(Config.LATEST_MODEL))
    ckpt = torch.load(Config.LATEST_MODEL, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    global_steps = ckpt.get("timesteps", 0)

    print(f"[INFO] Loaded model | steps={global_steps}")
else:
    print("[INFO] Training from scratch")

trainer = TRPO(model)
manager = ModelManager()

for ep in range(Config.EPISODES):

    states, actions, rewards, probs, dones = [], [], [], [], []

    total_steps = 0
    total_reward = 0

    while total_steps < Config.BATCH_SIZE:

        state = env.reset()

        while True:

            s = torch.tensor(state, dtype=torch.float32)

            action, prob = model.get_action(s)

            next_state, reward, done = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            probs.append(prob.numpy())
            dones.append(1 if done else 0)

            state = next_state

            total_reward += reward
            total_steps += 1

            if done or total_steps >= Config.BATCH_SIZE:
                break

    global_steps += 1
    kl = trainer.update(states, actions, probs, rewards, dones, global_steps)
    # ⭐ 保存模型
    manager.save_latest(model, global_steps)
    manager.update_best(model, total_reward)

    print(f"[TRPO] Ep {ep} | Reward {total_reward:.2f} | KL {kl:.5f}")