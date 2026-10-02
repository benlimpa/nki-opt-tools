"""Run a benchmark kernel against a torch reference on a *generated* shape.

This is approach (a) from the design discussion: instead of relying on each oracle's
hardcoded `run_test` shapes, we reuse the oracle's `_make_inputs` / `_ref` helpers but
feed them coverage-generated shapes, then compare the kernel `wrapper`'s output to the
torch/numpy reference. The oracle stays the bug detector; coverage chooses the shapes.

Targets the uniform nkibench + compiler_gen oracle convention:
    _make_inputs(<dim args...>, seed=42) -> tensor | tuple[tensor, ...]
    _ref(*numpy_arrays)                  -> numpy array | tensor
    wrapper(*tensors)                    -> tensor   (compared as .float())

Kernels are imported in-process; this module MUST run under the Neuron venv with the
device available (NEURON_PLATFORM_TARGET_OVERRIDE=trn1). It does NOT import torch at
module load, so the offline test suite can import it to exercise the pure helpers
(signature introspection, shape derivation).
"""
from __future__ import annotations

import importlib.util
import inspect
import pathlib
import sys
from dataclasses import dataclass

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


# ---------------------------------------------------------------- module loading

def _load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kernel_and_oracle_paths(source: str, kernel: str) -> tuple[pathlib.Path, pathlib.Path]:
    from triton_nki_transpile.bench_runs import common as c  # lazy: stdlib-only
    return c.kernel_path_for(source, kernel), c.oracle_path_for(source, kernel)


# ---------------------------------------------------------------- shape introspection

def make_inputs_dims(oracle_mod) -> list[str]:
    """Positional dim parameter names of `_make_inputs` (excluding `seed`).

    For the nkibench convention these are the shape knobs (M, K, N, B, D, S, ...).
    Returns [] for the compiler_gen convention (single `shapes` arg) — handled
    separately by the campaign, since those kernels are shape-specialized."""
    fn = getattr(oracle_mod, "_make_inputs", None)
    if fn is None:
        return []
    params = list(inspect.signature(fn).parameters.values())
    dims = [p.name for p in params
            if p.name != "seed" and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    # The compiler_gen convention takes a single structured `shapes` arg.
    if dims == ["shapes"]:
        return []
    return dims


def seed_shapes(oracle_mod) -> list[tuple]:
    """Shape tuples the oracle's own run_test exercises — parsed from the
    `for <dims> in [(...), ...]` literal in run_test. These are the known-good
    seeds the generator mutates from."""
    import ast
    src = inspect.getsource(oracle_mod.run_test)
    tree = ast.parse(src.lstrip())
    shapes: list[tuple] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.iter, (ast.List, ast.Tuple)):
            for elt in node.iter.elts:
                if isinstance(elt, ast.Tuple):
                    try:
                        shapes.append(tuple(ast.literal_eval(e) for e in elt.elts))
                    except (ValueError, SyntaxError):
                        pass
    return shapes


# ---------------------------------------------------------------- execution (needs device)

@dataclass
class OracleResult:
    source: str
    kernel: str
    shape: tuple
    ok: bool
    max_abs_err: float | None
    detail: str

    def __str__(self) -> str:
        e = "n/a" if self.max_abs_err is None else f"{self.max_abs_err:.4g}"
        return f"{self.kernel}{self.shape}: {'PASS' if self.ok else 'FAIL'} (err={e}) {self.detail}"


def run_shape(source: str, kernel: str, shape: tuple, *,
              atol: float = 1e-2, rtol: float = 1e-2, seed: int = 42) -> OracleResult:
    """Run `wrapper(*_make_inputs(*shape))` and compare to `_ref`. Imports torch
    and the kernel — must run under the Neuron venv with the device available."""
    import torch  # noqa: PLC0415
    from triton_nki_transpile.utils import extract_wrapper_name  # noqa: PLC0415

    kpath, opath = kernel_and_oracle_paths(source, kernel)
    kmod = _load_module(kpath, f"_k_{source}_{kernel}")
    omod = _load_module(opath, f"_o_{source}_{kernel}")

    wrapper_name = extract_wrapper_name(kpath.read_text())
    wrapper = getattr(kmod, wrapper_name)

    try:
        inputs = omod._make_inputs(*shape, seed=seed)
        if not isinstance(inputs, (tuple, list)):
            inputs = (inputs,)
        ref = omod._ref(*[t.numpy() for t in inputs])
        ref_t = torch.as_tensor(ref).float()
        out = wrapper(*inputs).float()
        if tuple(out.shape) != tuple(ref_t.shape):
            return OracleResult(source, kernel, shape, False, None,
                                f"shape mismatch out={tuple(out.shape)} ref={tuple(ref_t.shape)}")
        err = (out - ref_t).abs().max().item()
        ok = err <= atol + rtol * ref_t.abs().max().item()
        return OracleResult(source, kernel, shape, ok, err, "" if ok else "*** WRONG OUTPUT ***")
    except Exception as e:  # noqa: BLE001
        return OracleResult(source, kernel, shape, False, None, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- before/after pair (needs device)

@dataclass
class PairResult:
    """Three differentials for one (run pair, shape). Policy-free: the campaign
    applies the equivalence-class-appropriate tolerance to decide pass/fail.

    - before_vs_after: the real optimization differential (baseline vs optimized
      output on the SAME input). For bit-exact / fp-reorder opts this is the
      assertion; for precision opts (bf16 cast) it diverges by design and the
      torch anchors below are used instead.
    - before_vs_ref / after_vs_ref: each kernel vs the torch/numpy reference.
    """
    source: str
    kernel: str
    shape: tuple
    before_vs_after: float | None
    before_vs_ref: float | None
    after_vs_ref: float | None
    detail: str
    ref_absmax: float | None = None     # max|ref|, for magnitude-scaled tolerance

    @property
    def ran(self) -> bool:
        return self.before_vs_after is not None

    def _f(self, x):
        return "n/a" if x is None else f"{x:.4g}"

    def __str__(self) -> str:
        return (f"{self.kernel}{self.shape}: "
                f"b<->a={self._f(self.before_vs_after)} "
                f"b<->ref={self._f(self.before_vs_ref)} "
                f"a<->ref={self._f(self.after_vs_ref)}"
                + (f"  {self.detail}" if self.detail else ""))


def _max_abs_err(a, b) -> float | None:
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    if tuple(a.shape) != tuple(b.shape):
        return None
    return (a - b).abs().max().item()


def _build_conv_b(oracle_mod, recipe, shape: tuple, seed: int):
    """Convention B: build the helper-arg list and the torch reference for a
    generated shape by replaying the parsed run_test recipe. Returns
    (helper_args_factory, ref_tensor) where the factory yields a fresh arg list
    (so an in-place wrapper can't contaminate other runs)."""
    import torch  # noqa: PLC0415

    knobvals = dict(zip(recipe.knobs, shape))

    def _mk_tensor(t):
        dims = [knobvals[a] if isinstance(a, str) else a for a in t.axes]
        dtype = getattr(torch, t.dtype) if t.dtype else None
        kw = {"dtype": dtype} if dtype else {}
        if t.ctor == "full":
            return torch.full(tuple(dims), t.fill, **kw)
        if t.ctor in ("ones", "zeros", "randn", "rand", "empty"):
            return getattr(torch, t.ctor)(*dims, **kw)
        if t.ctor == "randint":
            return torch.randint(0, 100, tuple(dims), **kw)
        raise ValueError(f"unsupported ctor {t.ctor}")

    def build():
        torch.manual_seed(seed)
        tensors = {t.name: _mk_tensor(t) for t in recipe.tensors}
        def resolve(name):
            if name in tensors:
                return tensors[name]
            if name in recipe.scalars:
                return recipe.scalars[name]
            if name in knobvals:
                return knobvals[name]
            raise KeyError(name)
        return [resolve(n) for n in recipe.helper_args], tensors

    args, tensors = build()
    if recipe.ref_name is None:
        raise ValueError("recipe has no _ref function")
    ref_fn = getattr(oracle_mod, recipe.ref_name)
    ref_args = [tensors.get(n, recipe.scalars.get(n, knobvals.get(n))) for n in recipe.ref_args]
    ref = ref_fn(*ref_args)
    if isinstance(ref, dict):                      # some refs return {"output_0": ...}
        ref = next(iter(ref.values()))
    import torch  # noqa: PLC0415
    ref_t = torch.as_tensor(ref) if not hasattr(ref, "float") else ref
    return build, ref_t


def run_pair(baseline_path: pathlib.Path, optimized_path: pathlib.Path,
             oracle_path: pathlib.Path, shape: tuple, *,
             source: str = "", kernel: str = "", seed: int = 42) -> PairResult:
    """Run baseline and optimized kernels on the same generated shape and report
    all three differentials. Kernels run SEQUENTIALLY (never co-launched), so the
    multi-core constraint that rules out simultaneous launch does not apply.

    Inputs are regenerated per consumer (same seed -> identical values) so an
    in-place-writing wrapper cannot contaminate the other run or the reference."""
    import torch  # noqa: PLC0415
    from triton_nki_transpile.utils import extract_wrapper_name  # noqa: PLC0415

    bmod = _load_module(baseline_path, f"_base_{source}_{kernel}")
    omod_k = _load_module(optimized_path, f"_opt_{source}_{kernel}")
    oracle = _load_module(oracle_path, f"_orc_{source}_{kernel}")

    def _wrap(mod, path):
        return getattr(mod, extract_wrapper_name(path.read_text()))

    try:
        base_wrapper = _wrap(bmod, baseline_path)
        opt_wrapper = _wrap(omod_k, optimized_path)

        if hasattr(oracle, "_make_inputs"):
            # Convention A: factored _make_inputs(<dims>) + numpy _ref
            def _mk():
                ins = oracle._make_inputs(*shape, seed=seed)
                return ins if isinstance(ins, (tuple, list)) else (ins,)
            # _make_inputs may return non-tensor args (e.g. a Python float eps)
            # alongside tensors; convert only tensors, pass scalars through (as the
            # kernel's own run_test does).
            def _np(v):
                return v.numpy() if hasattr(v, "numpy") else v
            ref_t = torch.as_tensor(oracle._ref(*[_np(t) for t in _mk()])).float()
            before = base_wrapper(*_mk())
            after = opt_wrapper(*_mk())
        else:
            # Convention B: inline-shape oracle -> build from parsed recipe
            import oracle_recipe as orc  # noqa: PLC0415
            recipe = orc.parse_recipe(oracle_path.read_text())
            if recipe is None:
                return PairResult(source, kernel, shape, None, None, None,
                                  "no _make_inputs and no parseable recipe")
            build, ref_t = _build_conv_b(oracle, recipe, shape, seed)
            before = base_wrapper(*build()[0])
            after = opt_wrapper(*build()[0])

        return PairResult(
            source, kernel, shape,
            before_vs_after=_max_abs_err(before, after),
            before_vs_ref=_max_abs_err(before, ref_t),
            after_vs_ref=_max_abs_err(after, ref_t),
            detail="",
            ref_absmax=ref_t.detach().cpu().float().abs().max().item(),
        )
    except Exception as e:  # noqa: BLE001
        return PairResult(source, kernel, shape, None, None, None,
                          f"{type(e).__name__}: {e}")


def run_pair_for(run_pair_obj, shape: tuple, *, seed: int = 42) -> PairResult:
    """Convenience: run a run_index.RunPair on a shape."""
    return run_pair(run_pair_obj.baseline_path, run_pair_obj.optimized_path,
                    run_pair_obj.oracle_path, shape,
                    source=run_pair_obj.source, kernel=run_pair_obj.kernel, seed=seed)


def baseline_run_test_ok(carrier) -> bool | None:
    """Sanity gate: does the carrier's BASELINE pass its own oracle.run_test?
    True if it passes, False if it raises (broken-baseline run — its coverage
    verdicts are not trustworthy as findings), None if it can't be set up."""
    from triton_nki_transpile.utils import extract_wrapper_name  # noqa: PLC0415
    try:
        kmod = _load_module(carrier.baseline_path, f"_rtk_{carrier.source}_{carrier.kernel}")
        omod = _load_module(carrier.oracle_path, f"_rto_{carrier.source}_{carrier.kernel}")
        w = getattr(kmod, extract_wrapper_name(carrier.baseline_path.read_text()))
        run_test = getattr(omod, "run_test", None)
        if run_test is None:
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        run_test(w)
        return True
    except Exception:  # noqa: BLE001
        return False


def first_seed_shape(carrier) -> tuple | None:
    """The first shape the carrier's oracle tests (a known-good shape), in the
    grounding's knob order. Used to pin untiled knobs to a tested value so OOD is
    attributable to the tiled axis under test, not an arbitrary default."""
    try:
        omod = _load_module(carrier.oracle_path, f"_ss_{carrier.source}_{carrier.kernel}")
    except Exception:  # noqa: BLE001
        return None
    if hasattr(omod, "_make_inputs"):
        ss = seed_shapes(omod)
        return ss[0] if ss else None
    import oracle_recipe as orc  # noqa: PLC0415
    r = orc.parse_recipe(carrier.oracle_path.read_text())
    return r.seed_shapes[0] if (r and r.seed_shapes) else None
