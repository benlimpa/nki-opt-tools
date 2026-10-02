"""nki-bench: benchmark a kernel against its logic file's bench() function.

Targets:
  nki     — profile the compiled NEFF locally with neuron-profile
  triton  — send bench request to the Triton server (GPU)
  ref     — run _ref_*() from the logic file locally via time.perf_counter()

Usage examples:
  nki-bench --kernel-path kernels/nki/matmul.py \\
            --logic-path kernels/tests/logic/matmul_logic.py --target nki

  nki-bench --kernel-path kernels/triton/layernorm.py --func-name layer_norm \\
            --logic-path kernels/tests/logic/layernorm_logic.py --target triton

  nki-bench --logic-path kernels/tests/logic/matmul_logic.py --target ref
"""

import importlib.util
import json
import os
import time

import click
import zmq

from triton_nki_transpile.opt_profiler import NKIOptimizationProfiler
from triton_nki_transpile.utils import extract_wrapper_name


def _load_logic_module(logic_path: str):
    spec = importlib.util.spec_from_file_location("_logic", logic_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_ref_fn(mod):
    """Return the first callable whose name starts with '_ref_'."""
    for name in dir(mod):
        if name.startswith("_ref_") and callable(getattr(mod, name)):
            return getattr(mod, name)
    return None


def _bench_remote(kernel_path: str, func_name: str, logic_path: str,
                  addr: str, iters: int, warmup: int, timeout: int) -> dict:
    kernel_code = open(kernel_path).read()
    bench_code = open(logic_path).read()

    req = {
        "kernel_code": kernel_code,
        "execute_func_name": func_name,
        "mode": "bench",
        "bench_code": bench_code,
        "iters": iters,
        "warmup_iters": warmup,
        "seed": 0,
        "input_config": None,
        "runs": 1,
        "timeout": timeout,
    }

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(addr)
    sock.send_string(json.dumps(req))
    reply = sock.recv_string()
    sock.close()
    ctx.term()

    if reply.startswith("ERROR"):
        raise RuntimeError(reply)
    return json.loads(reply)


def _bench_ref(logic_path: str, iters: int, warmup: int) -> dict:
    mod = _load_logic_module(logic_path)
    if not hasattr(mod, "bench"):
        raise RuntimeError(f"Logic file {logic_path} has no bench() function")
    bench_fn = mod.bench

    ref_fn = _find_ref_fn(mod)
    if ref_fn is None:
        raise RuntimeError(f"Logic file {logic_path} has no _ref_* function")

    for _ in range(warmup):
        bench_fn(ref_fn)

    start = time.perf_counter()
    for _ in range(iters):
        bench_fn(ref_fn)
    elapsed = time.perf_counter() - start

    avg = elapsed / iters
    return {
        "total_time": elapsed,
        "avg_time_ms": avg * 1000,
        "iters": iters,
        "throughput_per_sec": 1.0 / avg if avg > 0 else 0,
    }


def _bench_nki_profiler(kernel_path: str, logic_path: str, iters: int, warmup: int, timeout: int) -> dict:
    kernel_code = open(kernel_path).read()
    bench_code = open(logic_path).read()

    wrapper_name = extract_wrapper_name(kernel_code)
    if not wrapper_name:
        raise RuntimeError(
            "Could not extract wrapper function name from kernel. "
            "Ensure the kernel has a '# Wrapper Function:' comment above the wrapper."
        )

    profiler = NKIOptimizationProfiler()
    num_exec = max(warmup + iters, 1)
    profile_nth = min(max(warmup + 1, 1), num_exec)
    result = profiler.profile_bench(
        kernel_code=kernel_code,
        bench_code=bench_code,
        wrapper_name=wrapper_name,
        timeout=timeout,
        num_exec=num_exec,
        profile_nth=profile_nth,
    )

    if not result["success"]:
        raise RuntimeError(result.get("error") or "NKI profiler failed")

    latency_ms = result.get("latency_ms")
    if latency_ms is None:
        raise RuntimeError(result.get("error") or "NKI profiler returned no latency")

    avg_s = latency_ms / 1000.0
    return {
        "total_time": avg_s * iters,
        "avg_time_ms": latency_ms,
        "iters": iters,
        "throughput_per_sec": 1.0 / avg_s if avg_s > 0 else 0,
    }


def _format_row(label: str, kernel_name: str, result: dict) -> str:
    avg_ms = result["avg_time_ms"]
    tps = result["throughput_per_sec"]
    return f"{kernel_name:<20s}  {label:<12s}  {avg_ms:>10.2f} ms/call  {tps:>10.2f} calls/s"


@click.command()
@click.option("--kernel-path", default=None, help="Path to kernel .py (not needed for --target ref)")
@click.option("--logic-path", required=True, help="Path to the *_logic.py file")
@click.option("--func-name", default=None, help="Wrapper function name in kernel file (required for --target triton)")
@click.option("--target", default="nki", type=click.Choice(["nki", "triton", "ref"]), show_default=True,
              help="Where to run the benchmark")
@click.option("--iters", default=50, show_default=True, help="Number of timed iterations")
@click.option("--warmup", default=3, show_default=True, help="Number of warmup iterations")
@click.option("--timeout", default=300, show_default=True, help="Seconds before the profiler or server kills a stalled kernel")
@click.option("--nki-addr", default=None, envvar="NKI_ADDR", help="Deprecated: unused for --target nki")
@click.option("--triton-addr", default=None, envvar="TRITON_ADDR", help="ZMQ address of Triton server")
@click.option("--json-out", is_flag=True, help="Print results as JSON")
@click.option("--kernel-name", default=None, help="Override the kernel name in JSON output (default: derived from logic file name)")
def main(kernel_path, logic_path, func_name, target, iters, warmup, timeout,
         nki_addr, triton_addr, json_out, kernel_name):
    """Benchmark a kernel using the bench() function from its logic file."""
    if kernel_name is None:
        kernel_name = os.path.splitext(os.path.basename(logic_path))[0].replace("_logic", "")

    if target == "ref":
        result = _bench_ref(logic_path, iters, warmup)
        label = "ref"
    else:
        if not kernel_path:
            raise click.UsageError("--kernel-path is required for --target nki/triton")

        if target == "nki":
            result = _bench_nki_profiler(kernel_path, logic_path, iters, warmup, timeout)
            label = "nki"
        else:
            if not func_name:
                raise click.UsageError("--func-name is required for --target triton")
            addr = triton_addr
            if not addr:
                addr_file = os.path.join(os.path.dirname(__file__), "../../../.triton_addr")
                if os.path.exists(addr_file):
                    addr = open(addr_file).read().strip()
            if not addr:
                raise click.UsageError("Triton server address not set. Use --triton-addr or $TRITON_ADDR")
            result = _bench_remote(kernel_path, func_name, logic_path, addr, iters, warmup, timeout)
            label = "triton"

    if json_out:
        result["kernel"] = kernel_name
        result["target"] = label
        print(json.dumps(result))
    else:
        print(_format_row(label, kernel_name, result))


if __name__ == "__main__":
    main()
