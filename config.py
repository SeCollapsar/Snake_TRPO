class Config:
    """
    全局配置文件（所有超参数统一管理）
    """

    # ---------- 训练 ----------
    EPISODES = 50000
    BATCH_SIZE = 3000

    # ---------- 模型 ----------
    MODEL_DIR = "models"
    LATEST_MODEL = MODEL_DIR + "/latest.pt"
    TOP_K_MODELS = 3

    # ---------- TRPO ----------
    TRPO_MAX_KL = 0.01
    TRPO_DAMPING = 0.1
    TRPO_LAMBDA = 0.95
    TRPO_GAMMA = 0.99
    TRPO_CG_ITERS = 10

    # ---------- 优化 ----------
    VALUE_LR = 1e-3

    # ---------- 环境 ----------
    GRID_SIZE = 10
    MAX_STEPS_WITHOUT_FOOD = 150

    # ---------- 奖励 ----------
    REWARD_DEATH = -50
    REWARD_EAT = 10
    REWARD_WIN = 50
    REWARD_STEP = -0.01
    REWARD_DISTANCE_FACTOR = 0.1
    REWARD_REPEAT_PENALTY = -0.02

    # ---------- 其他 ----------
    EPSILON = 0
    TEMPERATURE = 1.5

    # ---------- GUI ----------
    WINDOW_SIZE = 600
    FPS = 80

    # ---------- 动作 ----------
    ACTIONS = 4

    # ---------- 方向 ----------
    DIRECTIONS = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1)
    ]

    OPPOSITE = {
        0: 1,
        1: 0,
        2: 3,
        3: 2
    }