from env.snake_env import SnakeEnv
from rl.trpo.trpo_network import TRPONetwork
from rl.trpo.trpo_agent import TRPOAgent
from utils.logger_trpo import TRPOLogger
from config import Config

env = SnakeEnv()
net = TRPONetwork()
net.load()

agent = TRPOAgent(net)
logger = TRPOLogger()

best_reward = -1e9

for ep in range(Config.EPISODES):

    state = env.reset()

    states, actions, rewards, probs, dones = [], [], [], [], []

    total_reward = 0

    while True:

        action, prob = agent.sample_action(state)

        next_state, reward, done = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        probs.append(prob)
        dones.append(1 if done else 0)

        state = next_state
        total_reward += reward

        if done:
            break

    # ⭐ TRPO更新
    KL = agent.update(states, actions, probs, rewards, dones)

    net.save()

    if total_reward > best_reward:
        best_reward = total_reward

    logger.log(total_reward)

    if ep % 100 == 0:
        print('============================================')
        print(f"[TRPO] Episode {ep}, Reward: {total_reward}")
        print(f"KL Divergence: {KL}")
        print('============================================')
        

    if ep % 100 == 0:
        logger.save()