from __future__ import annotations

import os


def maybe_sanitize_cuda_runtime_env() -> None:
    flag = os.environ.get("DEEPVBH_SANITIZE_CUDA_RUNTIME", "")
    if flag.lower() not in {"1", "true", "yes", "on"}:
        return
    os.environ.pop("LD_LIBRARY_PATH", None)
