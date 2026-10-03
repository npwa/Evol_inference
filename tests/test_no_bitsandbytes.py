"""The search, fitness, probes and tests must import on machines without bitsandbytes (Arm Linux, macOS):
bitsandbytes is only needed by the CUDA path (`weight_bank`, GPU tests). Regression guard for a bug found
on the first Graviton run, where five test modules failed to collect on a machine without it."""

import subprocess
import sys

MODULES = [
    "evol_inference.genome", "evol_inference.ga", "evol_inference.baselines", "evol_inference.fitness",
    "evol_inference.validation", "evol_inference.search", "evol_inference.mo_ga", "evol_inference.mo_fitness",
    "evol_inference.gguf_assembler", "evol_inference.llama_probes", "evol_inference.sensitivity",
    "evol_inference.tabulated", "evol_inference.dryrun_setup", "evol_inference.platform_info",
]


def test_core_modules_import_without_bitsandbytes():
    code = (
        "import sys\n"
        "sys.modules['bitsandbytes'] = None  # makes `import bitsandbytes` raise ImportError\n"
        f"for m in {MODULES!r}:\n"
        "    __import__(m)\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-600:]
