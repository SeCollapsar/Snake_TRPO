import sys
import numpy as np
import torch

from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPainter, QColor, QFont
from PyQt6.QtCore import QTimer

from env.snake_env import SnakeEnv
from rl.trpo.trpo_model import ActorCritic
from config import Config


class SnakeWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.env = SnakeEnv()

        # ⭐ 模型
        input_dim = Config.GRID_SIZE * Config.GRID_SIZE
        self.policy = ActorCritic(input_dim, Config.ACTIONS)

        try:
            ckpt = torch.load(Config.LATEST_MODEL, map_location="cpu")
            self.policy.load_state_dict(ckpt["model"])
        except:
            print("[WARN] No model found")

        self.policy.eval()

        self.size = Config.GRID_SIZE
        self.window_size = Config.WINDOW_SIZE
        self.cell = self.window_size // self.size

        self.setWindowTitle("RL Snake (TRPO Torch)")
        self.setFixedSize(self.window_size, self.window_size)

        self.timer = QTimer()
        self.timer.timeout.connect(self.game_step)
        self.timer.start(Config.FPS)

        self.state = self.env.reset()

    def game_step(self):

        state_tensor = torch.tensor(self.state, dtype=torch.float32)

        with torch.no_grad():
            logits, _ = self.policy(state_tensor)
            probs = torch.softmax(logits, dim=-1).numpy()

        action = np.argmax(probs)

        self.state, _, done = self.env.step(action)

        if done:
            self.state = self.env.reset()

        self.update()

    def paintEvent(self, event):

        painter = QPainter(self)

        # ---------- 背景 ----------
        painter.fillRect(0, 0, self.window_size, self.window_size, QColor(30, 30, 30))

        # ---------- 网格 ----------
        painter.setPen(QColor(50, 50, 50))
        for i in range(self.size):
            painter.drawLine(0, i * self.cell, self.window_size, i * self.cell)
            painter.drawLine(i * self.cell, 0, i * self.cell, self.window_size)

        # ---------- 蛇 ----------
        for i, (x, y) in enumerate(self.env.snake):

            color = QColor(0, 255, 0) if i == 0 else QColor(0, 150, 0)

            painter.fillRect(
                y * self.cell,
                x * self.cell,
                self.cell,
                self.cell,
                color
            )

        # ---------- 食物 ----------
        if self.env.food:
            fx, fy = self.env.food
            painter.fillRect(
                fy * self.cell,
                fx * self.cell,
                self.cell,
                self.cell,
                QColor(255, 60, 60)
            )

        # ---------- 分数 ----------
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 14))

        score = len(self.env.snake) - 2
        painter.drawText(10, 20, f"Score: {score}")

        # ---------- 概率 ----------
        state_tensor = torch.tensor(self.state, dtype=torch.float32)

        with torch.no_grad():
            logits, _ = self.policy(state_tensor)
            probs = torch.softmax(logits, dim=-1).numpy()

        painter.drawText(10, 40, f"Up:    {probs[0]:.2f}")
        painter.drawText(10, 60, f"Down:  {probs[1]:.2f}")
        painter.drawText(10, 80, f"Left:  {probs[2]:.2f}")
        painter.drawText(10, 100, f"Right: {probs[3]:.2f}")


def run():
    app = QApplication(sys.argv)
    win = SnakeWindow()
    win.show()
    sys.exit(app.exec())