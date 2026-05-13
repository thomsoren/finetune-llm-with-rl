"""rl_finetuning: SPO advantage estimator + SOAP optimizer + math rewards.

Importing this package registers the SPO advantage estimator with verl.
SOAP is loaded by verl's pluggable optimizer system on demand; nothing to
register here. The math reward is wired via verl's data.custom_reward_function
config — no import-time hook needed.
"""

from . import spo  # noqa: F401  (side effect: register_adv_est("spo"))

__all__ = ["spo", "soap", "rewards"]
