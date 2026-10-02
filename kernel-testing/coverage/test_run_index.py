"""Offline tests for run_index — pure stdlib, no torch/device.

Builds a synthetic runs/ tree in a temp dir so the parsing/inversion logic is
exercised without depending on the real benchmark-opt/runs corpus.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import run_index as ri  # noqa: E402


def _make_run(root: pathlib.Path, name: str, *,
              opt_ids=None, with_pair=True, with_analysis=True):
    d = root / name
    (d / "workspace").mkdir(parents=True)
    if with_pair:
        (d / "workspace" / "kernel.py").write_text("def k(): pass\n")
        (d / "workspace" / "optimized_kernel.py").write_text("def k(): pass\n")
        (d / "workspace" / "oracle.py").write_text("def _ref(): pass\n")
    if with_analysis:
        payload = {"analysis": {"optimizations_selected_ids": opt_ids,
                                "final_optimization_successful": True}}
        (d / "analysis.json").write_text(json.dumps(payload))
    return d


# ---- parse_run_name ---------------------------------------------------------

def test_parse_run_name_ok():
    m = ri.parse_run_name("20260427_232017__nkibench__adamw__pre_explored@v1__codex_gpt54")
    assert m == {"source": "nkibench", "kernel": "adamw",
                 "approach": "pre_explored", "variant": "v1", "model": "codex_gpt54"}


def test_parse_run_name_no_at():
    m = ri.parse_run_name("20260427_232017__ref__matmul__baseline__model")
    assert m["approach"] == "baseline" and m["variant"] == ""


def test_parse_run_name_malformed():
    assert ri.parse_run_name("not__enough") is None


# ---- _flatten_opt_ids -------------------------------------------------------

def test_flatten_list_of_lists():
    assert ri._flatten_opt_ids([[31], [], [20]]) == [31, 20]


def test_flatten_dedupes_preserving_order():
    assert ri._flatten_opt_ids([[17], [17, 5], [5]]) == [17, 5]


def test_flatten_rejects_bools_and_nonints():
    assert ri._flatten_opt_ids([[True], ["x"], [12]]) == [12]


def test_flatten_none():
    assert ri._flatten_opt_ids(None) == []


# ---- index_runs -------------------------------------------------------------

def test_index_inverts_to_carriers():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _make_run(root, "T1__nkibench__adamw__pre@v1__m", opt_ids=[[31], [20]])
        _make_run(root, "T2__reference__matmul__pre@v1__m", opt_ids=[[31]])
        rep = ri.index_runs(root)
        assert set(rep.by_opt) == {31, 20}
        assert {f"{c.source}/{c.kernel}" for c in rep.by_opt[31]} == {
            "nkibench/adamw", "reference/matmul"}
        assert {f"{c.source}/{c.kernel}" for c in rep.by_opt[20]} == {"nkibench/adamw"}


def test_index_counts_unlabelled_pairs():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _make_run(root, "T1__nkibench__adamw__pre@v1__m", opt_ids=[[31]])
        _make_run(root, "T2__nkibench__bmm__pre@v1__m", with_analysis=False)
        rep = ri.index_runs(root)
        assert len(rep.labelled) == 1
        assert len(rep.unlabelled) == 1
        assert rep.unlabelled[0].kernel == "bmm"


def test_index_skips_labelled_without_pair_as_carrier():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _make_run(root, "T1__nkibench__adamw__pre@v1__m", opt_ids=[[31]], with_pair=False)
        rep = ri.index_runs(root)
        # labelled, but missing pair files -> not a usable carrier
        assert rep.by_opt == {}
        assert len(rep.labelled) == 1
        assert rep.labelled[0].has_pair is False


def test_index_reports_malformed_names():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "bad_name").mkdir()
        _make_run(root, "T1__nkibench__adamw__pre@v1__m", opt_ids=[[31]])
        rep = ri.index_runs(root)
        assert rep.malformed == ["bad_name"]
        assert len(rep.pairs) == 1


def test_empty_analysis_ids_not_a_carrier():
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        _make_run(root, "T1__nkibench__adamw__pre@v1__m", opt_ids=[[], []])
        rep = ri.index_runs(root)
        # analysis present (labelled flag) but no opt ids -> no carrier, not in labelled-with-ids
        assert rep.by_opt == {}
        assert len(rep.labelled) == 0


def _run() -> int:
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
