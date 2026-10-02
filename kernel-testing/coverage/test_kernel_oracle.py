"""Offline tests for kernel_oracle's pure introspection helpers.

These don't import torch/numpy or touch the device — they exercise the shape-derivation
logic against a synthetic oracle module, so they run in the plain (non-venv) interpreter.
The device-dependent `run_shape` is covered by the on-hardware campaign, not here.
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import kernel_oracle as ko  # noqa: E402

_NKIBENCH_SRC = (
    "def _make_inputs(M, K, N, seed=42):\n"
    "    return (M, K, N)\n"
    "def _ref(a, b):\n"
    "    return a\n"
    "def run_test(wrapper):\n"
    "    for M, K, N in [(128, 128, 512), (256, 256, 512)]:\n"
    "        pass\n"
)

_COMPILER_GEN_SRC = (
    "def _make_inputs(shapes, seed=42):\n"
    "    return shapes\n"
    "def _ref(*args):\n"
    "    return args\n"
    "def run_test(wrapper):\n"
    "    inputs = _make_inputs([[1, 2]])\n"
)


def _load_src(src: str, name: str):
    # seed_shapes uses inspect.getsource, which needs a real file on disk.
    tmp = pathlib.Path(tempfile.gettempdir()) / f"{name}.py"
    tmp.write_text(src)
    return ko._load_module(tmp, name)


def _fake_nkibench_oracle():
    return _load_src(_NKIBENCH_SRC, "fake_nkibench_oracle")


def _fake_compiler_gen_oracle():
    return _load_src(_COMPILER_GEN_SRC, "fake_cg_oracle")


def test_make_inputs_dims_nkibench():
    assert ko.make_inputs_dims(_fake_nkibench_oracle()) == ["M", "K", "N"]


def test_make_inputs_dims_compiler_gen_is_empty():
    # single structured `shapes` arg -> not a per-dim sweep
    assert ko.make_inputs_dims(_fake_compiler_gen_oracle()) == []


def test_seed_shapes_parsed_from_run_test():
    assert ko.seed_shapes(_fake_nkibench_oracle()) == [(128, 128, 512), (256, 256, 512)]


def test_seed_shapes_empty_when_no_for_literal():
    assert ko.seed_shapes(_fake_compiler_gen_oracle()) == []


# ---- PairResult (pure formatting; device path validated on hardware) ---------

def test_pairresult_ran_true_when_before_after_present():
    r = ko.PairResult("nkibench", "adamw", (128, 2048), 0.0, 4.8e-7, 4.8e-7, "")
    assert r.ran is True


def test_pairresult_ran_false_on_error():
    r = ko.PairResult("nkibench", "adamw", (128, 1024), None, None, None, "ValueError: x")
    assert r.ran is False


def test_pairresult_str_shows_all_three_differentials():
    r = ko.PairResult("nkibench", "adamw", (128, 1024), 0.0, 4.479, 4.479, "")
    s = str(r)
    assert "b<->a=0" in s and "b<->ref=4.479" in s and "a<->ref=4.479" in s


# ---- Convention-B input builder (needs torch; skipped without it) ------------

def test_build_conv_b_constructs_shapes_dtypes_and_ref():
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001
        print("  (skip: torch unavailable)")
        return
    import oracle_recipe as orc
    src = (
        "import torch\n"
        "def _ref_mm(a, b):\n    return torch.matmul(a.float(), b.float()).half()\n"
        "def run_test(helper):\n"
        "    for m, n, k in [(64, 64, 64)]:\n"
        "        a = torch.randn((m, k), dtype=torch.float16)\n"
        "        b = torch.randn((k, n), dtype=torch.float16)\n"
        "        expected = _ref_mm(a, b)\n"
        "        actual = helper(a, b)\n"
        "        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)\n"
    )
    recipe = orc.parse_recipe(src)
    mod = ko._load_src(src, "fake_b_oracle") if hasattr(ko, "_load_src") else None
    # load the module so _ref_mm is callable
    import pathlib, tempfile
    p = pathlib.Path(tempfile.gettempdir()) / "fake_b_oracle.py"
    p.write_text(src)
    mod = ko._load_module(p, "fake_b_oracle")
    # shape in knob order (m, n, k) = (32, 48, 16): a=(m,k)=(32,16), b=(k,n)=(16,48)
    build, ref = ko._build_conv_b(mod, recipe, (32, 48, 16), seed=0)
    args, tensors = build()
    import torch
    assert [tuple(t.shape) for t in args] == [(32, 16), (16, 48)]
    assert all(t.dtype == torch.float16 for t in args)
    assert tuple(ref.shape) == (32, 48)            # m x n


def test_build_conv_b_regenerates_identical_inputs():
    try:
        import torch
    except Exception:  # noqa: BLE001
        print("  (skip: torch unavailable)")
        return
    import oracle_recipe as orc, pathlib, tempfile
    src = ("import torch\n"
           "def _ref(x):\n    return x\n"
           "def run_test(helper):\n"
           "    for m, n in [(8, 8)]:\n"
           "        x = torch.randn((m, n))\n"
           "        expected = _ref(x)\n"
           "        actual = helper(x)\n"
           "        torch.testing.assert_close(actual, expected)\n")
    p = pathlib.Path(tempfile.gettempdir()) / "fake_b2.py"
    p.write_text(src)
    mod = ko._load_module(p, "fake_b2")
    recipe = orc.parse_recipe(src)
    build, _ = ko._build_conv_b(mod, recipe, (8, 8), seed=7)
    a1 = build()[0][0]
    a2 = build()[0][0]
    assert bool((a1 == a2).all())   # same seed -> identical, regenerable per consumer


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
