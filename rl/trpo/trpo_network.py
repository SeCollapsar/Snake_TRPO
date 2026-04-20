import numpy as np
import os
from config import Config


class TRPONetwork:

    def __init__(self, input_dim=100, hidden=128, output=4):

        # ---------- Policy ----------
        self.w1 = np.random.randn(input_dim, hidden) * 0.01
        self.b1 = np.zeros(hidden)

        self.w2 = np.random.randn(hidden, output) * 0.01
        self.b2 = np.zeros(output)

        # ---------- Value ----------
        self.w1_v = np.random.randn(input_dim, hidden) * 0.01
        self.b1_v = np.zeros(hidden)

        self.w2_v = np.random.randn(hidden, 1) * 0.01
        self.b2_v = np.zeros(1)

    def forward(self, x):

        # Policy
        h = np.tanh(np.dot(x, self.w1) + self.b1)
        logits = np.dot(h, self.w2) + self.b2
        probs = self.softmax(logits)

        # Value
        h_v = np.tanh(np.dot(x, self.w1_v) + self.b1_v)
        value = np.dot(h_v, self.w2_v) + self.b2_v

        return probs, h, value[0], h_v

    def softmax(self, x):
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    # ===== 参数处理 =====
    def get_params(self):
        return np.concatenate([
            self.w1.flatten(), self.b1,
            self.w2.flatten(), self.b2
        ])

    def set_params(self, flat):
        p = 0

        def get(shape):
            nonlocal p
            size = np.prod(shape)
            val = flat[p:p+size].reshape(shape)
            p += size
            return val

        self.w1 = get(self.w1.shape)
        self.b1 = get(self.b1.shape)
        self.w2 = get(self.w2.shape)
        self.b2 = get(self.b2.shape)

    # ===== 保存 =====
    def save(self):
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        np.save(f"{Config.MODEL_DIR}/trpo.npy", self.__dict__)

    def load(self):
        path = f"{Config.MODEL_DIR}/trpo.npy"
        if os.path.exists(path):
            self.__dict__.update(np.load(path, allow_pickle=True).item())