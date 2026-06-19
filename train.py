import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

from bullet_dodge_env.env import BulletEnv
from bullet_dodge_env.callbacks import CustomCheckpointCallback, CustomLogCallback
from bullet_dodge_env.utils import custom_linear_schedule
from cnn import CustomCombinedExtractor

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--checkpoint-freq", type=int, default=200_000)
    parser.add_argument("--resume", default=None, help="从已有模型继续训练，如 models/example_model.zip")
    args = parser.parse_args()

    env = BulletEnv(render_enabled=False, stack_size=4)

    lr_schedule = custom_linear_schedule(
        initial_lr=3e-4,
        final_lr=3e-5,
        total_steps=args.total_steps,
        decay_steps=500_000,
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    if args.resume:
        model = PPO.load(args.resume, env=env, tensorboard_log=args.log_dir)
        print(f"已加载模型: {args.resume}")
    else:
        model = PPO(
            policy="MlpPolicy",
            policy_kwargs=policy_kwargs,
            env=env,
            tensorboard_log=args.log_dir,
            verbose=1,
            learning_rate=lr_schedule,
            n_steps=1024,
            batch_size=512,
            n_epochs=10,
            gamma=0.95,
            gae_lambda=0.75,
            clip_range=0.1,
            ent_coef=0.005,
        )

    callbacks = CallbackList([
        CustomLogCallback(),
        CustomCheckpointCallback(save_freq=args.checkpoint_freq),
    ])
    model.learn(total_timesteps=args.total_steps, callback=callbacks, reset_num_timesteps=not args.resume)


if __name__ == "__main__":
    main()
