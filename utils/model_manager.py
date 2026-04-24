import torch
import os
from datetime import datetime
from config import Config


class ModelManager:

    def __init__(self):
        self.model_dir = Config.MODEL_DIR
        self.top_k = Config.TOP_K_MODELS

        os.makedirs(self.model_dir, exist_ok=True)

        # [(reward, state_dict)]
        self.best_models = []

    # =========================
    # 保存当前模型
    # =========================
    def save_latest(self, model, total_timesteps):

        torch.save({
            "model": model.state_dict(),
            "timesteps": total_timesteps
        }, Config.LATEST_MODEL)

    # =========================
    # Top-K插入逻辑（核心）
    # =========================
    def update_best(self, model, reward):

        import re
        import os
        from datetime import datetime

        # =========================
        # 1. 读取历史 best 文件
        # =========================
        files = [f for f in os.listdir(self.model_dir) if f.startswith("best-")]

        history = []

        for f in files:
            # 从文件名提取 reward
            match = re.search(r"\(([-\d\.]+)\)", f)
            if match:
                r = float(match.group(1))
                history.append((r, f))

        # 按 reward 排序（从高到低）
        history.sort(key=lambda x: x[0], reverse=True)

        # =========================
        # 2. 判断是否进入 TOP-3
        # =========================
        if len(history) >= self.top_k:
            worst_reward = history[-1][0]
            if reward <= worst_reward:
                return  # ❌ 不进入TOP-3，直接退出

        # =========================
        # 3. 保存新模型（命名保持一致）
        # =========================
        date_str = datetime.now().strftime("%d-%m")

        temp_name = f"best-temp-({round(reward, 2)})--{date_str}.pt"
        temp_path = os.path.join(self.model_dir, temp_name)

        torch.save(self._clone_state_dict(model), temp_path)

        history.append((reward, temp_name))
        history.sort(key=lambda x: x[0], reverse=True)

        # =========================
        # 4. 保留 TOP-3
        # =========================
        topk = history[:self.top_k]

        keep_files = set([f for _, f in topk])

        # 删除非TOP-3
        for f in files:
            if f not in keep_files:
                os.remove(os.path.join(self.model_dir, f))

        # =========================
        # 5. 重命名为标准格式（保持你要求）
        # =========================
        for i, (r, f) in enumerate(topk):
            old_path = os.path.join(self.model_dir, f)
            new_name = f"best-{i+1}-({round(r, 2)})--{date_str}.pt"
            new_path = os.path.join(self.model_dir, new_name)

            os.rename(old_path, new_path)
    # =========================
    # 深拷贝模型
    # =========================
    def _clone_state_dict(self, model):
        return {k: v.clone() for k, v in model.state_dict().items()}