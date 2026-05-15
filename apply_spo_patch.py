"""Patch upstream verl so Ray workers can lazy-load our custom adv estimator
and the SPO-chain MC continuation agent loop.

Why this is needed:
- verl's adv estimators and agent loops are registered via @register decorators
  at module-import time. Built-in registrations happen when verl imports
  `core_algos.py` and `agent_loop.py`. Our custom registrations live under
  `rl_finetuning.*`; the driver imports `rl_finetuning` via `rl_finetuning.train`,
  but Ray *actors* (rollout workers) are separate Python processes that import
  verl modules directly and never touch `rl_finetuning`. Without help they hit:
    - "Unknown advantage estimator: spo"  (driver-side, on first training step)
    - "Agent loop spo_mc_agent not registered" (worker-side, when MC rollouts run)

What we (don't) try first:
- A .pth file that `import rl_finetuning` at interpreter startup also pulls
  torch in via the verl chain BEFORE Ray sets CUDA_VISIBLE_DEVICES per worker,
  which collapses the 4-rank -> 4-GPU mapping (NCCL "Duplicate GPU detected").
- Ray runtime_env injection - verl owns the ray.init() call so we can't pass
  in py_modules / env_vars cleanly without forking verl harder.

What this patch does:
1. In `core_algos.get_adv_estimator_fn`: on a registry miss, import the package
   named by env var $VERL_ADV_EST_USER_PKG. Runs late, AFTER Ray has bound the
   worker to its assigned GPU, so torch initializes with the right CUDA_VISIBLE_DEVICES.
2. In `agent_loop.py`: at the end of module load, attempt to import the same
   package. This makes any @register agent loops in that package available in
   worker processes. Wrapped in try/except so worker boot never crashes if the
   package or its deps aren't importable.

Set `VERL_ADV_EST_USER_PKG=rl_finetuning` in the row scripts.

Idempotent. Run after `bash scripts/install_vllm_sglang_mcore.sh`.
"""

import os
import py_compile

REPO = os.environ.get("VERL_REPO", "/workspace/verl")
CORE_ALGOS = f"{REPO}/verl/trainer/ppo/core_algos.py"
AGENT_LOOP = f"{REPO}/verl/experimental/agent_loop/agent_loop.py"

CORE_ALGOS_MARKER = "VERL_ADV_EST_USER_PKG"
CORE_ALGOS_ORIG = (
    "    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum\n"
    "    if name not in ADV_ESTIMATOR_REGISTRY:\n"
    "        raise ValueError(f\"Unknown advantage estimator simply: {name}\")\n"
    "    return ADV_ESTIMATOR_REGISTRY[name]"
)
CORE_ALGOS_PATCHED = (
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

AGENT_LOOP_MARKER = "# VERL_ADV_EST_USER_PKG hook (rl_finetuning patch)"
AGENT_LOOP_APPEND = (
    "\n\n" + AGENT_LOOP_MARKER + "\n"
    "import os as _spo_os\n"
    "_spo_pkg = _spo_os.environ.get(\"VERL_ADV_EST_USER_PKG\")\n"
    "if _spo_pkg:\n"
    "    try:\n"
    "        __import__(_spo_pkg)\n"
    "    except Exception:\n"
    "        pass\n"
)


def patch_file(path: str, marker: str, original: str | None, patched: str | None, *, append: str | None = None) -> bool:
    with open(path) as f:
        src = f.read()
    if marker in src:
        print(f"{os.path.basename(path)}: already patched")
        return False
    if append is not None:
        new_src = src + append
    else:
        assert original is not None and patched is not None
        if original not in src:
            print(
                f"{path}: ERROR - expected block not found.\n"
                f"  verl version may have changed; inspect {path} manually."
            )
            raise SystemExit(1)
        new_src = src.replace(original, patched)
    with open(path, "w") as f:
        f.write(new_src)
    py_compile.compile(path, doraise=True)
    print(f"{os.path.basename(path)}: patched")
    return True


def main():
    patch_file(CORE_ALGOS, CORE_ALGOS_MARKER, CORE_ALGOS_ORIG, CORE_ALGOS_PATCHED)
    patch_file(AGENT_LOOP, AGENT_LOOP_MARKER, None, None, append=AGENT_LOOP_APPEND)


if __name__ == "__main__":
    main()
