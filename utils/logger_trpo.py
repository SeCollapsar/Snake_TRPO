import os
import matplotlib.pyplot as plt


class TRPOLogger:

    def __init__(self):
        self.rewards = []
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, reward):
        self.rewards.append(reward)

    def save(self):

        plt.figure()
        plt.plot(self.rewards)
        plt.title("TRPO Reward Curve")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.savefig(os.path.join(self.log_dir, "trpo_reward.png"))
        plt.close()