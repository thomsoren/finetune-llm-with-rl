"""rl_finetuning: SPO advantage estimator + SOAP/Kron optimizers + math rewards.

Importing this package:
  - registers the SPO advantage estimator with verl,
  - registers the "spo_mc_agent" agent loop (used by SPO-chain MC rollouts).

SOAP/Kron are loaded by verl's pluggable optimizer system on demand. The math
reward is wired via verl's data.custom_reward_function config — no import-time
hook needed.
"""

from . import spo  # noqa: F401  (side effect: register_adv_est("spo"))
from . import spo_mc_agent  # noqa: F401  (side effect: register agent loop)

__all__ = ["spo", "soap", "kron", "rewards", "spo_mc", "spo_mc_agent", "spo_trainer"]
