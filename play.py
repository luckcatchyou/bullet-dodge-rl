import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bullet_dodge_env.env import Game


if __name__ == "__main__":
    Game(render_enabled=True, debug_obs=False).play()
