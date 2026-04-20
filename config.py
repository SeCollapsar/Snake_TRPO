class Config:
    """
    全局配置文件（所有超参数统一管理）
    """
    # ---------- 训练 ----------
    EPISODES =                  20000

    # ---------- 模型 ----------
    MODEL_DIR = "models"
    MAX_BACKUP_MODELS =         10

    # ---------- 环境 ----------
    GRID_SIZE =                 10
    MAX_STEPS_WITHOUT_FOOD =    150

    # ---------- 奖励 ----------
    REWARD_DEATH =              -50
    REWARD_EAT =                15
    REWARD_WIN =                50
    REWARD_STEP =               -0.01
    REWARD_DISTANCE_FACTOR =    0.1
    REWARD_REPEAT_PENALTY =     -0.05

    # ---------- 其他 ----------
    EPSILON =                   0
    TEMPERATURE =               1.5

    # ---------- GUI ----------
    WINDOW_SIZE =               600
    FPS =                       80  # ms（越小越快）

    # ---------- 动作 ----------
    ACTIONS =                   4

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