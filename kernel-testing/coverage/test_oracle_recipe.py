"""Offline tests for oracle_recipe.parse_recipe — pure stdlib, no torch."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import oracle_recipe as orc  # noqa: E402

_MATMUL = (
    "import torch\n"
    "def _ref_matmul(a, b):\n    return torch.matmul(a.float(), b.float()).half()\n"
    "def run_test(helper_func):\n"
    "    shapes = [(64, 64, 64), (128, 128, 128)]\n"
    "    for m, n, k in shapes:\n"
    "        a = torch.randn((m, k), dtype=torch.float16)\n"
    "        b = torch.randn((k, n), dtype=torch.float16)\n"
    "        expected = _ref_matmul(a, b)\n"
    "        actual = helper_func(a, b)\n"
    "        torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)\n"
)
_LAYERNORM = (
    "import torch\n"
    "def _ref_layer_norm(x, w, b, eps):\n    return x\n"
    "def run_test(helper_func):\n"
    "    eps = 1e-5\n"
    "    for m, n in [(1, 128), (4, 256)]:\n"
    "        x = torch.randn((m, n), dtype=torch.float32)\n"
    "        w = torch.randn((n,), dtype=torch.float32)\n"
    "        b = torch.randn((n,), dtype=torch.float32)\n"
    "        expected = _ref_layer_norm(x, w, b, eps)\n"
    "        actual = helper_func(x, w, b, eps)\n"
    "        torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)\n"
)
_NO_RECIPE = (
    "import numpy as np, torch\n"
    "def _make_input(b, h):\n    return torch.tensor(np.random.randn(b, h))\n"
    "def run_test(wrapper):\n"
    "    for b, h in [(1, 256)]:\n"
    "        x = _make_input(b, h)\n"
    "        wrapper(x)\n"
)


def test_matmul_recipe():
    r = orc.parse_recipe(_MATMUL)
    assert r.knobs == ["m", "n", "k"]
    assert r.seed_shapes[:2] == [(64, 64, 64), (128, 128, 128)]
    assert {(t.name, tuple(t.axes), t.dtype) for t in r.tensors} == {
        ("a", ("m", "k"), "float16"), ("b", ("k", "n"), "float16")}
    assert r.helper_args == ["a", "b"]
    assert r.ref_name == "_ref_matmul" and r.ref_args == ["a", "b"]
    assert r.atol == 1e-2 and r.rtol == 1e-2


def test_layernorm_recipe_has_1d_tensors_and_scalar():
    r = orc.parse_recipe(_LAYERNORM)
    assert r.knobs == ["m", "n"]
    tb = r.tensor_by_name
    assert tb["w"].axes == ["n"] and tb["x"].axes == ["m", "n"]
    assert r.helper_args == ["x", "w", "b", "eps"]      # eps is a scalar arg
    assert r.scalars.get("eps") == 1e-5


def test_no_recipe_when_inputs_not_inline():
    assert orc.parse_recipe(_NO_RECIPE) is None


def test_scalar_excludes_knob_names():
    # `n = 128` later in run_test must not shadow knob n as a scalar
    src = _MATMUL.replace(
        "        torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)\n",
        "        torch.testing.assert_close(actual.cpu(), expected, atol=1e-2, rtol=1e-2)\n"
        "    n = 128\n")
    r = orc.parse_recipe(src)
    assert "n" not in r.scalars


def test_assert_close_tol_extracted():
    assert orc.assert_close_tol(_MATMUL) == (1e-2, 1e-2)
    assert orc.assert_close_tol(_LAYERNORM) == (1e-3, 1e-3)


def test_assert_close_tol_absent():
    assert orc.assert_close_tol("def f():\n    pass\n") == (None, None)


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
