"""Smoke test for a Colab session: prove we got an L4 and that torch can use it.

Run with:  colab exec -s <session> -f scripts/smoke_l4.py

It asserts the GPU is an L4 rather than printing and moving on, because the CLI
silently falls back to A100 when `--gpu` is not recognised, and this programme
is L4-only by instruction. A wrong GPU here is cheaper to catch than 24h later.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: no usable nvidia-smi ({exc})")
        return 1

    print(f"nvidia-smi: {smi}")

    try:
        import torch
    except ImportError:
        print("FAIL: torch is not importable on this VM")
        return 1

    print(f"torch {torch.__version__}, cuda {torch.version.cuda}, "
          f"available={torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("FAIL: torch cannot see the GPU")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"device: {name}  capability sm_{cap[0]}{cap[1]}  {total_gb:.1f} GB")

    # A real allocation and matmul: "available" has lied before on a VM whose
    # GPU was already claimed by a zombie kernel.
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    import time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = 50 * 2 * 4096 ** 3 / dt / 1e12
    print(f"fp16 matmul: {tflops:.1f} TFLOP/s over {dt:.2f}s")

    # bf16 works on Ada (sm_89); worth knowing before a run picks a dtype.
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    if "L4" not in name:
        print(f"FAIL: expected an L4, got {name!r}. This programme is L4-only; "
              f"stop this session and re-create it with --gpu L4.")
        return 1

    print("OK: L4 confirmed and usable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
