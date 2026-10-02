"""nki-profile: Profile an NKI kernel against its oracle test.

Usage:
    nki-profile --kernel-path kernel.py --oracle-path oracle.py [--wrapper-name my_op]
"""
import json
import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

from triton_nki_transpile.opt_profiler import NKIOptimizationProfiler
from triton_nki_transpile.utils import extract_wrapper_name

load_dotenv()
logging.basicConfig(level=logging.WARNING)


@click.command()
@click.option("--kernel-path", required=True, type=click.Path(exists=True), help="NKI kernel Python file")
@click.option("--oracle-path", required=True, type=click.Path(exists=True), help="Oracle test file defining run_test(wrapper) and optionally bench(wrapper)")
@click.option("--wrapper-name", default=None, help="Wrapper function name (auto-detected if omitted)")
@click.option("--timeout", default=300, show_default=True, help="Profiling timeout in seconds")
@click.option("--bench", "use_bench", is_flag=True, default=True, help="Profile using bench() instead of run_test() for benchmark-shaped inputs")
@click.option("--bench3", "use_bench3", is_flag=True, default=False, help="Profile 3 input shapes via bench3(wrapper, idx) and report mean latency (overrides --bench)")
def main(kernel_path: str, oracle_path: str, wrapper_name: str | None, timeout: int, use_bench: bool, use_bench3: bool):
    """Profile an NKI kernel and print JSON result to stdout."""
    kernel_code = Path(kernel_path).read_text()
    test_code = Path(oracle_path).read_text()

    if not wrapper_name:
        wrapper_name = extract_wrapper_name(kernel_code)
    if not wrapper_name:
        print(json.dumps({"success": False, "latency_ms": None, "error": "Could not determine wrapper name; pass --wrapper-name"}))
        sys.exit(0)

    profiler = NKIOptimizationProfiler()
    if use_bench3:
        result = profiler.profile_bench3(kernel_code, test_code, wrapper_name, timeout=timeout)
    elif use_bench:
        result = profiler.profile_bench(kernel_code, test_code, wrapper_name, timeout=timeout)
    else:
        result = profiler.profile(kernel_code, test_code, wrapper_name, timeout=timeout)
    # Flatten hardware_metrics into the top-level output for easy agent consumption
    hw = result.pop("hardware_metrics", None)
    if hw:
        # latency_ms is already at top level; merge remaining metrics
        for k, v in hw.items():
            if k != "latency_ms":
                result[k] = v
    print(json.dumps(result))


if __name__ == "__main__":
    main()
