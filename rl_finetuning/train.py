"""Entry point: import our pkg (registers SPO + MC agent loop), install the
SPO-chain trainer patches, then hand off to verl's PPO main.

Use as:
    python -m rl_finetuning.train  algorithm.adv_estimator=...  data.train_files=...  ...
"""

import rl_finetuning  # noqa: F401  (registers SPO adv estimator + MC agent loop)
from rl_finetuning.spo_trainer import install as install_spo_patches

install_spo_patches()

from verl.trainer.main_ppo import main  # noqa: E402


if __name__ == "__main__":
    main()
