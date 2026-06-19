import os

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class CustomLogCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.max_steps = 0

    def _on_step(self) -> bool:
        max_episode_steps = self.training_env.get_attr("max_episode_steps")[0]
        infos = self.locals["infos"]

        for info in infos:
            self.max_steps = max(self.max_steps, info.get("steps", 0))

        if max_episode_steps != float("inf") and self.num_timesteps % max_episode_steps == 0:
            self.logger.record("custom/max_steps", self.max_steps)
            self.max_steps = 0

        return True


class CustomCheckpointCallback(CheckpointCallback):
    def __init__(self, save_freq):
        super().__init__(save_freq=save_freq, save_path="", name_prefix="")

    def _init_callback(self) -> None:
        log_dir = self.logger.get_dir() or "logs"
        self.save_path = os.path.join(log_dir, "checkpoints")
        super()._init_callback()
