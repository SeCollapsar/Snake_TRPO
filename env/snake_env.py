import numpy as np
import random
import time

from config import Config


class SnakeEnv:

    def __init__(self):

        self.size = Config.GRID_SIZE
        self.max_cells = self.size * self.size

        self.max_steps_without_food = Config.MAX_STEPS_WITHOUT_FOOD

        self.directions = Config.DIRECTIONS
        self.opposite = Config.OPPOSITE

        self.reset()

    def reset(self):

        mid = self.size // 2

        self.snake = [
            (mid, mid),
            (mid, mid - 1)
        ]

        self.current_action = 3

        self.score = 0
        self.done = False

        self.steps_since_last_food = 0

        self.visited = {}

        self.spawn_food()

        return self.get_state()

    def spawn_food(self):

        random.seed(time.time_ns())

        empty = []

        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in self.snake:
                    empty.append((x, y))

        if len(empty) == 0:
            self.food = None
            return

        self.food = random.choice(empty)

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, action):

        if action == self.opposite[self.current_action]:
            action = self.current_action

        self.current_action = action

        head = self.snake[0]

        old_dist = self.manhattan(head, self.food)

        move = self.directions[action]
        new_head = (head[0] + move[0], head[1] + move[1])

        # ---------- 撞墙 ----------
        if (
            new_head[0] < 0
            or new_head[0] >= self.size
            or new_head[1] < 0
            or new_head[1] >= self.size
        ):
            return self.get_state(), Config.REWARD_DEATH, True

        # ---------- 撞身体 ----------
        if new_head in self.snake:
            return self.get_state(), Config.REWARD_DEATH, True

        self.snake.insert(0, new_head)

        reward = Config.REWARD_STEP

        new_dist = self.manhattan(new_head, self.food)

        reward += Config.REWARD_DISTANCE_FACTOR * (old_dist - new_dist)

        self.steps_since_last_food += 1

        # ---------- 重复路径惩罚 ----------
        if new_head not in self.visited:
            self.visited[new_head] = 0

        self.visited[new_head] += 1

        repeat_penalty = Config.REWARD_REPEAT_PENALTY * np.log(self.visited[new_head] + 1)
        reward += repeat_penalty

        # ---------- 吃到食物 ----------
        if new_head == self.food:

            reward = Config.REWARD_EAT

            self.steps_since_last_food = 0

            # 👉 清空访问记录（关键）
            self.visited = {}

            if len(self.snake) == self.max_cells:
                return self.get_state(), reward + Config.REWARD_WIN, True

            self.spawn_food()

        else:
            self.snake.pop()

        # ---------- 超时 ----------
        if self.steps_since_last_food >= self.max_steps_without_food:
            return self.get_state(), Config.REWARD_DEATH, True

        return self.get_state(), reward, False

    def get_state(self):

        grid = np.zeros((3, self.size, self.size))

        for x, y in self.snake:
            grid[0, x, y] = 1

        if self.food is not None:
            fx, fy = self.food
            grid[1, fx, fy] = 1

        hx, hy = self.snake[0]
        grid[2, hx, hy] = 1

        return grid.flatten()