"""Patch upstream verl so Ray workers can lazy-load our custom adv estimator.

Why this is needed:
- verl's adv estimators are registered via @register_adv_est at module-import
  time, into ADV_ESTIMATOR_REGISTRY in verl/trainer/ppo/core_algos.py.
- Built-ins (grpo, gae, rloo, ...) register themselves when verl imports
  core_algos. Our SPO lives in `rl_finetuning.spo`; the driver process imports
  it via `rl_finetuning.train`, but Ray *actors* are separate Python processes
  that import verl.trainer.main_ppo directly and never load our package, so
  `algorithm.adv_estimator=spo` blows up at the first training step with:
      ValueError: Unknown advantage estimator simply: spo

What we (don't) try first:
- A .pth file that `import rl_finetuning` at interpreter startup also pulls
  torch in via the verl chain BEFORE Ray sets CUDA_VISIBLE_DEVICES per worker,
  which collapses the 4-rank → 4-GPU mapping (NCCL "Duplicate GPU detected").
- Ray runtime_env injection — verl owns the ray.init() call so we can't pass
  in py_modules / env_vars cleanly without forking verl harder.

What this patch does:
- In `get_adv_estimator_fn`: on a registry miss, import the package named by
  the env var $VERL_ADV_EST_USER_PKG (default unset). The import runs late,
  AFTER Ray has bound the worker to its assigned GPU, so torch initializes
  with the correct CUDA_VISIBLE_DEVICES.
- Set `VERL_ADV_EST_USER_PKG=rl_finetuning` in the row scripts.

Idempotent. Run after `bash scripts/install_vllm_sglang_mcore.sh`.
"""
import os
import py_compile

REPO = os.environ.get("VERL_REPO", "/workspace/verl")
TARGET = f"{REPO}/verl/trainer/ppo/core_algos.py"

MARKER = "VERL_ADV_EST_USER_PKG"
ORIG = (
    "    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum\n"
    "    if name not in ADV_ESTIMATOR_REGISTRY:\n"
    "        raise ValueError(f\"Unknown advantage estimator simply: {name}\")\n"
    "    return ADV_ESTIMATOR_REGISTRY[name]"
)
PATCHED = (
    "    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum\n"
    "    if name not in ADV_ESTIMATOR_REGISTRY:\n"
    "        import os\n"
    "        pkg = os.environ.get(\"VERL_ADV_EST_USER_PKG\")\n"
    "        if pkg:\n"
    "            try:\n"
    "                __import__(pkg)\n"
    "            except ImportError:\n"
    "                pass\n"
    "    if name not in ADV_ESTIMATOR_REGISTRY:\n"
    "        raise ValueError(f\"Unknown advantage estimator simply: {name}\")\n"
    "    return ADV_ESTIMATOR_REGISTRY[name]"
)


def main():
    with open(TARGET) as f:
        src = f.read()

    if MARKER in src:
        print(f"core_algos.py: already patched")
        return

    if ORIG not in src:
        print(
            f"core_algos.py: ERROR - expected block not found at insertion point.\n"
            f"  verl version may have changed. Inspect get_adv_estimator_fn in {TARGET}."
        )
        raise SystemExit(1)

    with open(TARGET, "w") as f:
        f.write(src.replace(ORIG, PATCHED))
    print(f"core_algos.py: patched")

    py_compile.compile(TARGET, doraise=True)
    print(f"core_algos.py: compiles OK")


if __name__ == "__main__":
    main()
