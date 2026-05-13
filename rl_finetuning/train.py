"""Entry point: import our pkg (registers SPO), then hand off to verl's PPO main.

Use as:
    python -m rl_finetuning.train  algorithm.adv_estimator=...  data.train_files=...  ...
"""

import rl_finetuning  # noqa: F401  (registers SPO via side effect)
from verl.trainer.main_ppo import main


if __name__ == "__main__":
    main()
