"""Offline tests for grounding — pure stdlib, no torch/device.

Synthetic sources exercise the AST logic; a real-carrier smoke test confirms the
name-join works on the actual nkibench corpus (grounding reads source text only,
so it runs without the Neuron venv).
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import grounding as gr  # noqa: E402

_ORACLE_MN = (
    "import torch\n"
    "def _make_inputs(M, N, seed=42):\n"
    "    return torch.randn(M, N)\n"
    "def _ref(a):\n"
    "    return a\n"
)
_ORACLE_SHAPES = (
    "def _make_inputs(shapes, seed=42):\n"
    "    return shapes\n"
)
_ORACLE_NONE = "def _ref(a):\n    return a\n"

# kernel tiling N by a named constant; M is not tiled by a divisor
_KERNEL_NTILE = (
    "N_TILE = 2048\n"
    "P = 128\n"
    "def adamw(theta, g, m, v):\n"
    "    M, N = theta.shape\n"
    "    n_f_tiles = N // N_TILE\n"
    "    return theta\n"
)
# optimized: same knob tiled by a *different* constant -> union
_KERNEL_NTILE_OPT = (
    "N_TILE = 512\n"
    "def adamw(theta, g, m, v):\n"
    "    M, N = theta.shape\n"
    "    n_f_tiles = N // N_TILE\n"
    "    return theta\n"
)
# kernel tiling a knob by an int literal directly
_KERNEL_LITERAL = (
    "def k(x):\n"
    "    M, N = x.shape\n"
    "    t = M % 128\n"
    "    return x\n"
)


# ---- oracle_knobs -----------------------------------------------------------

def test_oracle_knobs_mn():
    assert gr.oracle_knobs(_ORACLE_MN) == ["M", "N"]


def test_oracle_knobs_shapes_is_none():
    assert gr.oracle_knobs(_ORACLE_SHAPES) is None


def test_oracle_knobs_absent_is_none():
    assert gr.oracle_knobs(_ORACLE_NONE) is None


# ---- tiling_divisors --------------------------------------------------------

def test_tiling_divisors_named_const():
    assert gr.tiling_divisors(_KERNEL_NTILE, ["M", "N"]) == {"N": {2048}}


def test_tiling_divisors_literal_mod():
    assert gr.tiling_divisors(_KERNEL_LITERAL, ["M", "N"]) == {"M": {128}}


def test_tiling_divisors_ignores_non_knob_numerator():
    # `n_f_tiles // 2` must not register: n_f_tiles is not a knob
    src = "def k(x):\n    M, N = x.shape\n    a = N // 512\n    b = n_f_tiles // 2\n    return x\n"
    assert gr.tiling_divisors(src, ["M", "N"]) == {"N": {512}}


# ---- ground_sources (partition) ---------------------------------------------

def test_ground_unions_baseline_and_optimized_divisors():
    g = gr.ground_sources("nkibench", "adamw", _ORACLE_MN,
                          _KERNEL_NTILE, _KERNEL_NTILE_OPT)
    assert g.tile_divisors == {"N": [512, 2048]}


def test_ground_emits_three_classes_per_divisor():
    g = gr.ground_sources("nkibench", "adamw", _ORACLE_MN,
                          _KERNEL_NTILE, _KERNEL_NTILE)  # single divisor 2048
    names = {c.name for c in g.classes}
    assert names == {"N_x2048_exact", "N_x2048_partial", "N_x2048_below"}
    crit = {c.name for c in g.classes if c.critical}
    assert crit == {"N_x2048_partial", "N_x2048_below"}


def test_ground_classes_partition_the_axis():
    # exactly one class matches any positive N
    g = gr.ground_sources("nkibench", "adamw", _ORACLE_MN, _KERNEL_NTILE, _KERNEL_NTILE)
    cls = [c for c in g.classes if c.divisor == 2048]
    for N in [1, 1024, 2048, 2049, 4096, 5000]:
        hits = [c.name for c in cls if eval(c.predicate, {}, {"N": N})]
        assert len(hits) == 1, f"N={N} hit {hits}"


def test_ground_ungrounded_when_no_knob_convention():
    g = gr.ground_sources("nl_to_nki", "rms", _ORACLE_NONE, "def k(): pass", "def k(): pass")
    assert g.grounded is False
    assert any("_make_inputs" in u for u in g.ungrounded)


def test_ground_ungrounded_when_no_tiling_over_knobs():
    g = gr.ground_sources("x", "k", _ORACLE_MN,
                          "def k(x):\n    M, N = x.shape\n    return x\n",
                          "def k(x):\n    M, N = x.shape\n    return x\n")
    assert g.grounded is False
    assert any("tiling" in u for u in g.ungrounded)


# ---- real-carrier smoke (source-only, no venv) ------------------------------

# ---- ceil-div / helper / product-divisor extraction (style) ----------------

def test_tiling_ceil_add_is_pad():
    src = ("TILE_M = 128\n"
           "def k(x):\n    M, N = x.shape\n    nt = (M + TILE_M - 1) // TILE_M\n    return x\n")
    assert gr.tiling_facts(src, ["M", "N"]) == {"M": {128: "pad"}}


def test_tiling_math_ceil_is_pad():
    src = ("import math\nTILE_M = 128\n"
           "def k(x):\n    M, N = x.shape\n    p = math.ceil(M / TILE_M)\n    return x\n")
    assert gr.tiling_facts(src, ["M", "N"]) == {"M": {128: "pad"}}


def test_tiling_div_ceil_helper_is_pad():
    src = ("COL_TILE = 512\n"
           "def div_ceil(n, d):\n    return (n + d - 1) // d\n"
           "def k(x):\n    R, C = x.shape\n    t = div_ceil(C, COL_TILE)\n    return x\n")
    assert gr.tiling_facts(src, ["R", "C"]) == {"C": {512: "pad"}}


def test_tiling_product_divisor_resolves():
    src = ("TILE_N = 512\nNUM_CORES = 2\nimport math\n"
           "def k(x):\n    M, N = x.shape\n    p = math.ceil(N / (TILE_N * NUM_CORES))\n    return x\n")
    assert gr.tiling_facts(src, ["M", "N"]) == {"N": {1024: "pad"}}


def test_tiling_computed_constant_resolves():
    # BLOCK_M = TILE_M * TILES_IN_BLOCK_M  must resolve for `M // BLOCK_M`
    src = ("TILE_M = 128\nTILES_IN_BLOCK_M = 4\nBLOCK_M = TILE_M * TILES_IN_BLOCK_M\n"
           "def k(x):\n    M, N = x.shape\n    nb = M // BLOCK_M\n    return x\n")
    assert gr.tiling_facts(src, ["M", "N"]) == {"M": {512: "drop"}}


def test_bare_floordiv_still_drop():
    src = "N_TILE=2048\ndef k(x):\n    M, N = x.shape\n    t = N // N_TILE\n    return x\n"
    assert gr.tiling_facts(src, ["M", "N"]) == {"N": {2048: "drop"}}


def test_drop_wins_over_pad_on_merge():
    base = "T=128\ndef k(x):\n    M,N=x.shape\n    a=M//T\n    return x\n"          # drop
    opt = ("import math\nT=128\ndef k(x):\n    M,N=x.shape\n"
           "    a=math.ceil(M/T)\n    return x\n")                                   # pad
    g = gr.ground_sources("s", "k", _ORACLE_MN, base, opt)
    # M tiled by 128, drop wins -> partial/below critical
    crit = {c.name for c in g.classes if c.critical}
    assert crit == {"M_x128_partial", "M_x128_below"}


def test_pad_style_classes_not_critical():
    src = ("import math\nT=128\n"
           "def k(x):\n    M,N=x.shape\n    a=math.ceil(M/T)\n    return x\n")
    g = gr.ground_sources("s", "k", _ORACLE_MN, src, src)
    assert g.grounded
    assert [c for c in g.classes if c.critical] == []   # ceil+pad -> no critical


def test_real_nkibench_matmul_grounds():
    import run_index as ri
    rep = ri.index_runs()
    cs = [c for o in rep.by_opt for c in rep.by_opt[o]
          if c.source == "nkibench" and c.kernel == "matmul"]
    if not cs:
        print("  (skip: no nkibench/matmul carrier present)")
        return
    g = gr.ground_carrier(cs[0])
    assert g.grounded, g.summary()
    # matmul tiles M,K by P=128 and N by N_TILE=512 -> all three knobs grounded
    assert set(g.tile_divisors) == {"M", "K", "N"}, g.tile_divisors


# ---- Convention-B structural join ------------------------------------------

_ORACLE_B_MM = (
    "import torch\n"
    "def _ref_mm(a, b):\n    return a\n"
    "def run_test(helper):\n"
    "    for m, n, k in [(64, 64, 64)]:\n"
    "        a = torch.randn((m, k))\n"
    "        b = torch.randn((k, n))\n"
    "        expected = _ref_mm(a, b)\n"
    "        actual = helper(a, b)\n"
    "        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)\n"
)
# wrapper takes (a, b, activation=""): a is (M,K) logically, tiled M//512, K//128;
# also builds lhsT = a.T (transposed) and unpacks K2,M2 = lhsT.shape — must be IGNORED.
_KERNEL_B_MM = (
    "BLOCK_M = 512\nTILE_K = 128\nTILE_N = 512\n"
    "def mm(a, b, activation=''):\n"
    "    M, K = a.shape\n"
    "    _, N = b.shape\n"
    "    nb_m = M // BLOCK_M\n"
    "    nb_k = K // TILE_K\n"
    "    nb_n = N // TILE_N\n"
    "    lhsT = a.T.contiguous()\n"
    "    K2, M2 = lhsT.shape\n"
    "    return a\n"
)


def test_structural_join_handles_transpose():
    g = gr.ground_sources("reference", "mm", _ORACLE_B_MM, _KERNEL_B_MM, _KERNEL_B_MM)
    assert g.grounded, g.summary()
    # a=(m,k) -> param a=(M,K): m=M (//512), k=K (//128); b=(k,n) -> n=N (//512)
    assert g.tile_divisors == {"m": [512], "k": [128], "n": [512]}, g.tile_divisors


def test_structural_refuses_when_only_transposed_unpack():
    # kernel exposes ONLY the transposed unpack (no direct a.shape) -> unmappable
    kernel = ("BLOCK_M=512\nTILE_K=128\n"
              "def mm(a, b):\n"
              "    lhsT = a.T.contiguous()\n"
              "    K, M = lhsT.shape\n"
              "    x = M // BLOCK_M\n    y = K // TILE_K\n    return a\n")
    g = gr.ground_sources("reference", "mm", _ORACLE_B_MM, kernel, kernel)
    # 'a'/'k'/'m' cannot be mapped to a kernel dim -> not grounded on those
    assert g.tile_divisors == {}, g.tile_divisors
    assert not g.grounded


def test_real_reference_matmul_structural():
    import run_index as ri
    rep = ri.index_runs()
    cs = [c for o in rep.by_opt for c in rep.by_opt[o]
          if c.source == "reference" and c.kernel == "matmul"]
    if not cs:
        print("  (skip: no reference/matmul carrier)")
        return
    g = gr.ground_carrier(cs[0])
    assert g.grounded, g.summary()
    # m = output rows (BLOCK_M=512), k = contraction (TILE_K=128) -- NOT swapped
    assert g.tile_divisors.get("m") == [512], g.tile_divisors
    assert g.tile_divisors.get("k") == [128], g.tile_divisors


def test_real_layernorm_grounds_pad_noncritical():
    import run_index as ri
    rep = ri.index_runs()
    cs = [c for o in rep.by_opt for c in rep.by_opt[o]
          if c.source == "nl_to_nki" and c.kernel == "layernorm"]
    if not cs:
        print("  (skip: no layernorm carrier)")
        return
    g = gr.ground_carrier(cs[0])
    assert g.grounded and g.tile_divisors.get("m") == [128], g.tile_divisors
    assert [c for c in g.classes if c.critical] == []   # ceil-div+pad -> no critical


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
