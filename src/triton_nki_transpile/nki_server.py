import importlib.util
import json
import multiprocessing
import os
import queue
import sys
import tempfile
import time
import uuid

import click
import numpy as np
import torch
import zmq

# Server config
PORT = 5555
DEFAULT_WORKER_TIMEOUT = 300  # seconds; overridable via $NKI_WORKER_TIMEOUT


def kernel_worker(kernel_code, execute_func_name, seed, input_config, runs, result_queue, mode="run", iters=10, test_code=None, bench_code=None, warmup_iters=3):
    temp_file_path = None
    module_name = f"kernel_{uuid.uuid4().hex}"
    try:
        print(f"--- SUBPROCESS START: {execute_func_name} (mode={mode}) ---")

        # Write kernel code to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".py", mode='w', delete=False) as f:
            f.write(kernel_code)
            temp_file_path = f.name

        # Load the temporary file as a module
        spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, execute_func_name):
            available = [k for k in dir(module) if callable(getattr(module, k))]
            raise KeyError(f"Function '{execute_func_name}' not found in module. Available: {available}")

        helper_func = getattr(module, execute_func_name)

        if mode == "bench":
            if not bench_code:
                raise ValueError("bench_code is required for mode='bench'")

            bench_ns = {}
            exec(bench_code, bench_ns)
            if "bench" not in bench_ns:
                raise KeyError("bench_code must define a 'bench' function")

            bench_fn = bench_ns["bench"]

            print(f"Warming up ({warmup_iters} iters)...")
            for _ in range(warmup_iters):
                bench_fn(helper_func)

            print(f"Benchmarking for {iters} iterations...")
            start_time = time.perf_counter()
            for _ in range(iters):
                bench_fn(helper_func)
            end_time = time.perf_counter()

            total_time = end_time - start_time
            avg_time = total_time / iters
            result_queue.put(("SUCCESS", {
                "total_time": total_time,
                "avg_time_ms": avg_time * 1000,
                "iters": iters,
                "throughput_per_sec": 1.0 / avg_time if avg_time > 0 else 0,
            }))
        elif mode == "test":
            if not test_code:
                raise ValueError("test_code is required for mode='test'")

            print("Executing unit test...")
            # Execute test_code to get run_test function
            test_ns = {}
            exec(test_code, test_ns)
            if "run_test" not in test_ns:
                raise KeyError("test_code must define a 'run_test' function")

            run_test = test_ns["run_test"]
            run_test(helper_func)
            print("Unit test passed!")
            result_queue.put(("SUCCESS", "Test passed"))
        elif mode == "profile":
            print("Warming up...")
            helper_func(input_config)

            print(f"Benchmarking for {iters} iterations...")
            start_time = time.perf_counter()
            for _ in range(iters):
                helper_func(input_config)
            end_time = time.perf_counter()

            total_time = end_time - start_time
            avg_time = total_time / iters
            result_queue.put(("SUCCESS", {
                "total_time": total_time,
                "avg_time": avg_time,
                "iters": iters,
                "throughput": 1.0 / avg_time if avg_time > 0 else 0,
            }))
        else:
            torch.manual_seed(seed)
            results = []
            for i in range(runs):
                output = helper_func(input_config)
                if isinstance(output, torch.Tensor):
                    output = output.cpu().numpy()
                results.append(output)

            arr = np.stack(results)
            result_queue.put(("SUCCESS", arr))

        print("--- SUBPROCESS FINISHED SUCCESSFULLY ---")
    except Exception as e:
        import traceback
        err_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"--- SUBPROCESS ERROR ---\n{err_msg}")
        result_queue.put(("ERROR", err_msg))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        if module_name in sys.modules:
            del sys.modules[module_name]


def run_server():
    worker_timeout = int(os.environ.get("NKI_WORKER_TIMEOUT", DEFAULT_WORKER_TIMEOUT))

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{PORT}")
    print(f"Starting NKI server on port {PORT} (worker timeout={worker_timeout}s)...")

    while True:
        message = socket.recv()
        req = json.loads(message.decode())

        kernel_code = req.get("kernel_code")
        execute_func_name = req.get("execute_func_name", "execute_kernel")
        seed = req.get("seed", 0)
        input_config = req.get("input_config")
        runs = req.get("runs", 1)
        mode = req.get("mode", "run")
        iters = req.get("iters", 10)
        test_code = req.get("test_code")
        bench_code = req.get("bench_code")
        warmup_iters = req.get("warmup_iters", 3)
        timeout = req.get("timeout", worker_timeout)

        print(f"Received request for {execute_func_name} (mode={mode}, timeout={timeout}s)")

        if not kernel_code:
            socket.send_string("ERROR: No kernel_code provided")
            continue

        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=kernel_worker,
            args=(kernel_code, execute_func_name, seed, input_config, runs, result_queue, mode, iters, test_code, bench_code, warmup_iters)
        )
        p.start()

        try:
            status, res = result_queue.get(timeout=timeout)
        except queue.Empty:
            print(f"ERROR: worker timed out after {timeout}s, killing subprocess")
            p.kill()
            p.join()
            socket.send_string(f"ERROR: kernel timed out after {timeout}s")
            continue

        p.join()

        if status == "ERROR":
            socket.send_string(f"ERROR: {res}")
        else:
            if mode in ("test",):
                socket.send_string("SUCCESS")
            elif mode in ("profile", "bench"):
                socket.send_string(json.dumps(res))
            else:
                socket.send(res.tobytes())


@click.command()
def main():
    run_server()


if __name__ == "__main__":
    main()
