"""Phase 0 sanity check: CUDA visibility and package imports (Doc/implementation_plan.md)."""

import torch
import transformers
import accelerate
import bitsandbytes
import datasets


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available to torch"
    device_name = torch.cuda.get_device_name(0)
    print(f"torch {torch.__version__} (cuda {torch.version.cuda})")
    print(f"transformers {transformers.__version__}")
    print(f"accelerate {accelerate.__version__}")
    print(f"bitsandbytes {bitsandbytes.__version__}")
    print(f"datasets {datasets.__version__}")
    print(f"device: {device_name}")


if __name__ == "__main__":
    main()
