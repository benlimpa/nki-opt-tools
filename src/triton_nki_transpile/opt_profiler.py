"""
NKI kernel profiler using neuron-profile capture + view for accurate hardware latency.

Approach (mirrors .agents/skills/nki-optimizer/scripts/profile_kernel.sh):
  Phase 1 — Run the kernel script in a temp work dir with NEURON_FRAMEWORK_DEBUG=1 and
             NEURON_PLATFORM_TARGET_OVERRIDE=trn1 so the compiler writes a NEFF into the
             work dir.  The most recently modified *.neff found there is used — this ties
             the NEFF to the specific kernel that was just compiled rather than pulling in
             every NEFF that happens to be in a shared cache.
  Phase 2 — Run `neuron-profile capture` + `neuron-profile view --output-format=summary-json`
             on that NEFF to extract the steady-state hardware latency.
"""
import json
import logging
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)


def _capture_nrt_diagnostics(child_env: dict) -> str:
    """Snapshot system state at the moment of an NRT 'Available:0' failure.

    Captured so we can finally figure out what's holding the cores when the
    nki-profile subprocess loses kernel[2]: parent + child env (NEURON_/PJRT_/NCCL_),
    device file owners (lsof /dev/neuron*), neuron-ls topology, and the visible
    python process tree. Best-effort: every sub-capture is wrapped so a missing
    binary or a hang doesn't mask the original error.
    """
    lines: list[str] = []
    relevant_prefixes = ("NEURON", "PJRT", "NCCL", "RT_", "XLA_", "TPU_")

    def _filtered_env(env: dict) -> str:
        return "\n".join(f"  {k}={v}" for k, v in sorted(env.items())
                         if any(k.startswith(p) for p in relevant_prefixes))

    lines.append(f"PID: parent={os.getpid()}")
    lines.append(f"CWD: {os.getcwd()}")
    lines.append("PARENT env (filtered):")
    lines.append(_filtered_env(os.environ))  # type: ignore[arg-type]
    lines.append("CHILD env (filtered):")
    lines.append(_filtered_env(child_env))

    def _run(label: str, cmd: list[str], timeout: int = 5) -> None:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            out = (p.stdout + p.stderr).strip()
            lines.append(f"--- {label} (rc={p.returncode}) ---")
            lines.append(out[:2000] if out else "(no output)")
        except FileNotFoundError:
            lines.append(f"--- {label} --- (binary not found)")
        except subprocess.TimeoutExpired:
            lines.append(f"--- {label} --- (timed out after {timeout}s)")
        except Exception as e:  # noqa: BLE001
            lines.append(f"--- {label} --- (capture failed: {type(e).__name__}: {e})")

    _run("ls -la /dev/neuron*", ["bash", "-lc", "ls -la /dev/neuron* 2>&1"])
    _run("lsof /dev/neuron0", ["bash", "-lc", "lsof /dev/neuron0 2>&1 || echo '(no lsof or no holders)'"])
    _run("neuron-ls", ["neuron-ls"])
    _run("python procs", ["bash", "-lc", "ps -eo pid,ppid,user,cmd | grep -E 'python|codex|claude' | grep -v grep | head -20"])
    return "\n".join(lines)


def _avg_numeric_fields(dicts: list[dict]) -> dict:
    """Mean of numeric values for keys present in every dict; non-numeric/missing keys dropped."""
    if not dicts:
        return {}
    common = set(dicts[0].keys())
    for d in dicts[1:]:
        common &= set(d.keys())
    out: dict = {}
    for k in common:
        vals = [d[k] for d in dicts]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            out[k] = round(sum(vals) / len(vals), 3)
    return out


def _build_profile_script(kernel_code: str, workload_code: str, run_block: str) -> str:
    """Return a self-contained Python script whose __main__ block exercises the kernel."""
    return f"""\
{kernel_code}

{workload_code}

if __name__ == "__main__":
{run_block}
"""


class NKIOptimizationProfiler:
    """
    Profiles NKI kernels using neuron-profile capture + view for accurate hardware latency.

    Returns {{"success": bool, "latency_ms": float|None, "error": str|None}}
    """

    def profile(
        self,
        kernel_code: str,
        test_code: str,
        wrapper_name: str,
        timeout: int = 300,
    ) -> dict:
        """
        Profile kernel correctness and hardware latency via run_test(wrapper).

        Args:
            kernel_code:  Full NKI kernel Python source.
            test_code:    Oracle test defining run_test(wrapper).
            wrapper_name: Name of the wrapper function in kernel_code.
            timeout:      Subprocess timeout in seconds.

        Returns:
            {"success": bool, "latency_ms": float|None, "error": str|None}
        """
        script = _build_profile_script(
            kernel_code=kernel_code,
            workload_code=test_code,
            run_block=f"""\
    run_test({wrapper_name})
    print("Test passed")""",
        )
        return self._run_profile(script, success_marker="Test passed", timeout=timeout)

    def profile_bench(
        self,
        kernel_code: str,
        bench_code: str,
        wrapper_name: str,
        timeout: int = 300,
        num_exec: int = 20,
        profile_nth: int = 10,
    ) -> dict:
        """
        Profile an NKI kernel using benchmark-shaped inputs from bench(wrapper).

        Returns {"success": bool, "latency_ms": float|None, "error": str|None}
        """
        script = _build_profile_script(
            kernel_code=kernel_code,
            workload_code=bench_code,
            run_block=f"""\
    if "bench" not in globals():
        raise KeyError("bench_code must define a 'bench' function")
    _out = bench({wrapper_name})
    import torch_xla; torch_xla.sync()
    print("Bench setup passed")""",
        )
        return self._run_profile(
            script,
            success_marker="Bench setup passed",
            timeout=timeout,
            num_exec=num_exec,
            profile_nth=profile_nth,
        )

    def profile_bench3(
        self,
        kernel_code: str,
        bench_code: str,
        wrapper_name: str,
        timeout: int = 300,
        num_exec: int = 20,
        profile_nth: int = 10,
    ) -> dict:
        """
        Profile an NKI kernel across 3 input shapes via bench3(wrapper, idx).

        Compiles + profiles each shape independently, then averages latency.
        Returns the same shape as profile_bench, with mean latency_ms,
        averaged numeric hardware_metrics, a per_case_latency_ms list, and
        per_case_results carrying each shape's full _run_profile result
        (including its individual hardware_metrics dict).
        On any failure, returns the failing case's result unchanged.
        """
        per_case: list[dict] = []
        for idx in (0, 1, 2):
            script = _build_profile_script(
                kernel_code=kernel_code,
                workload_code=bench_code,
                run_block=f"""\
    if "bench3" not in globals():
        raise KeyError("bench_code must define a 'bench3' function")
    _out = bench3({wrapper_name}, {idx})
    import torch_xla; torch_xla.sync()
    print("Bench setup passed")""",
            )
            result = self._run_profile(
                script,
                success_marker="Bench setup passed",
                timeout=timeout,
                num_exec=num_exec,
                profile_nth=profile_nth,
            )
            if not result.get("success") or result.get("latency_ms") is None:
                err = result.get("error") or f"bench3 case {idx} produced no latency"
                return {
                    "success": False,
                    "latency_ms": None,
                    "error": f"[bench3 idx={idx}] {err}",
                    "per_case_results": per_case + [result],
                }
            per_case.append(result)

        latencies = [r["latency_ms"] for r in per_case]
        avg_latency = round(sum(latencies) / len(latencies), 3)
        avg_metrics = _avg_numeric_fields([r.get("hardware_metrics") or {} for r in per_case])
        avg_metrics["latency_ms"] = avg_latency
        return {
            "success": True,
            "latency_ms": avg_latency,
            "error": None,
            "hardware_metrics": avg_metrics,
            "per_case_latency_ms": latencies,
            "per_case_results": per_case,
        }

    # ── Private helpers ──────────────────────────────────────────────────────

    def _run_profile(
        self,
        script: str,
        success_marker: str,
        timeout: int,
        num_exec: int = 20,
        profile_nth: int = 10,
    ) -> dict:
        """Compile the kernel, discover the generated NEFF, and profile it."""
        with tempfile.TemporaryDirectory(prefix="nki_opt_profile_") as tmp:
            tmp_path = pathlib.Path(tmp)
            # Run from a dedicated work dir so NEFF files don't scatter across the repo.
            # NEURON_FRAMEWORK_DEBUG=1 causes the compiler to write *.neff files here.
            work_dir = tmp_path / "work"
            work_dir.mkdir()

            script_file = tmp_path / "profile_kernel.py"
            script_file.write_text(script)

            # ── Phase 1: compile kernel ──────────────────────────────────────
            logger.info("Phase 1: Compiling kernel (generating NEFF)...")
            env = os.environ.copy()
            env["NEURON_FRAMEWORK_DEBUG"] = "1"
            env["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn1"
            try:
                proc = subprocess.run(
                    [sys.executable, str(script_file)],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                    cwd=str(work_dir),
                )
            except subprocess.TimeoutExpired:
                return {"success": False, "latency_ms": None, "error": "Phase 1 timed out"}

            stdout = proc.stdout
            stderr = proc.stderr

            if proc.returncode != 0:
                error_msg = stderr.strip() or stdout.strip()
                logger.warning(f"Phase 1 failed (rc={proc.returncode}): {error_msg[:2000]}")
                # Diagnostic capture for NRT 'Available:0' failures — this is the
                # nki-profile subprocess kernel[2] bug we're chasing. Capture
                # everything that might explain why allocation failed: env (NEURON_*),
                # device holders (lsof /dev/neuron*), process tree, neuron-ls.
                # See benchmark-opt/reports/mg_v2_giveup_traces.md.
                if "Logical Neuron Core(s) not available" in error_msg or "nrt_allocate_neuron_cores" in error_msg:
                    diag = _capture_nrt_diagnostics(child_env=env)
                    logger.warning("NRT-FAIL diagnostics:\n%s", diag)
                    error_msg = error_msg + "\n\n--- NRT-FAIL diagnostics ---\n" + diag
                return {"success": False, "latency_ms": None, "error": error_msg[:8000]}

            if success_marker not in stdout:
                logger.warning(
                    f"Phase 1 did not produce expected marker '{success_marker}'. "
                    f"stdout: {stdout[:200]}"
                )
                return {
                    "success": False,
                    "latency_ms": None,
                    "error": f"Expected '{success_marker}' not found in output",
                }

            # Find the most recently modified NEFF in the work dir (mirrors the
            # `find … -printf '%T@ %p\n' | sort -rn | head -1` logic in profile_kernel.sh).
            neff_files = list(work_dir.rglob("*.neff"))
            if not neff_files:
                logger.warning("No NEFF files found in work dir after Phase 1.")
                return {"success": True, "latency_ms": None, "error": "No NEFF found for profiling"}

            neff_path = max(neff_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found NEFF: {neff_path}")

            # Copy NEFF to a stable path so neuron-profile can always find it
            neff_dest = tmp_path / "kernel.neff"
            neff_dest.write_bytes(neff_path.read_bytes())

            # ── Phase 2: profile the NEFF ────────────────────────────────────
            logger.info("Phase 2: Profiling NEFF with neuron-profile...")
            metrics = self._benchmark_neff(
                str(neff_dest),
                tmp_path,
                num_exec=num_exec,
                profile_nth=profile_nth,
            )
            if metrics is None:
                return {"success": True, "latency_ms": None, "error": "neuron-profile failed"}

            lat = metrics.get("latency_ms")
            logger.info(f"Latency: {lat:.3f} ms")
            return {"success": True, "latency_ms": lat, "error": None, "hardware_metrics": metrics}

    def _benchmark_neff(
        self,
        neff_path: str,
        tmp_dir: pathlib.Path,
        num_exec: int = 20,
        profile_nth: int = 10,
    ) -> Optional[dict]:
        """
        Run neuron-profile capture + view on a NEFF.

        Returns a dict with latency_ms and selected hardware_metrics, or None on failure.
        --profile-nth-exec=N causes neuron-profile to write <base>_exec_N.ntff rather
        than <base>.ntff (mirrored from profile_kernel.sh step 2).
        """
        ntff_base = str(tmp_dir / "profile.ntff")

        capture_cmd = [
            "neuron-profile", "capture",
            "-n", neff_path,
            "-s", ntff_base,
            "--num-exec", str(num_exec),
            "--profile-nth-exec", str(profile_nth),
        ]
        try:
            p_cap = subprocess.run(
                capture_cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
        except subprocess.TimeoutExpired:
            logger.error("neuron-profile capture timed out")
            return None

        if p_cap.returncode != 0:
            logger.error(f"neuron-profile capture failed: {p_cap.stderr[:300]}")
            return None

        # Resolve the actual NTFF output path.
        # --profile-nth-exec=N writes <base>_exec_N.ntff (not <base>.ntff directly).
        ntff_stem = ntff_base.rsplit(".ntff", 1)[0]
        steady_ntff = f"{ntff_stem}_exec_{profile_nth}.ntff"
        if not os.path.exists(steady_ntff):
            steady_ntff = f"{ntff_stem}_exec_{num_exec}.ntff"
        if not os.path.exists(steady_ntff):
            logger.error(f"No NTFF file found after profiling (looked for _exec_{profile_nth} and _exec_{num_exec})")
            return None

        view_cmd = [
            "neuron-profile", "view",
            "-n", neff_path,
            "-s", steady_ntff,
            "--output-format=summary-json",
        ]
        try:
            p_view = subprocess.run(
                view_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            logger.error("neuron-profile view timed out")
            return None

        for line in p_view.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    key = list(data.keys())[0]
                    raw = data[key]
                    total_time_sec = raw.get("total_time", 0)
                    latency_ms = round(total_time_sec * 1000.0, 3)
                    metrics: dict = {"latency_ms": latency_ms}
                    for k, v in raw.items():
                        if k == "total_time":
                            continue  # already exposed as latency_ms
                        # Convert raw seconds time fields to ms for consistency
                        if k == "total_active_time":
                            metrics["total_active_time_ms"] = round(v * 1000.0, 3)
                        else:
                            metrics[k] = v
                    return metrics
                except (json.JSONDecodeError, IndexError, KeyError) as e:
                    logger.error(f"Failed to parse neuron-profile JSON: {e}")
                    return None

        logger.error("No JSON summary found in neuron-profile view output")
        return None
