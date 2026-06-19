import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bullet_dodge_env.env import BulletEnv


DEFAULT_MODEL = PROJECT_ROOT / "models" / "example_model.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--steps", type=int, default=100_000)
    args = parser.parse_args()

    env = BulletEnv(render_enabled=True, stack_size=4)
    model = PPO.load(args.model, env=env)
    obs, _ = env.reset()

    for _ in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
