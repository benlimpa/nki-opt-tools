import torch
import torch.nn.functional as F

def _ref_cross_entropy(logits, labels, logit_softcapping=0.0, logit_scaling=0.0):
    """PyTorch reference for the plain-CE case (no softcapping / no scaling)."""
    logits = logits.float()
    if logit_scaling != 0.0:
        logits = logit_scaling * logits
    if logit_softcapping != 0.0:
        logits = logit_softcapping * torch.tanh(logits / logit_softcapping)
    return F.cross_entropy(logits, labels.long(), reduction="none", ignore_index=-100)

def run_test(helper_func):
    """Property-based test for the cross_entropy kernel.

    ``helper_func(logits, labels)`` must accept CPU tensors and return a
    per-row loss tensor (possibly on device). Comparison done after .cpu().
    """
    vocab = 256

    # 1. Differential test – several batch sizes
    for batch in [1, 4, 16, 32]:
        logits = torch.randn((batch, vocab))
        labels = torch.randint(0, vocab, (batch,))
        expected = _ref_cross_entropy(logits, labels)
        actual = helper_func(logits, labels)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)

    # 2. Varying vocabulary sizes
    for v in [32, 64, 128, 512, 1024]:
        logits = torch.randn((8, v))
        labels = torch.randint(0, v, (8,))
        expected = _ref_cross_entropy(logits, labels)
        actual = helper_func(logits, labels)
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)

    # 3. Non-negativity: loss must be >= 0 for all valid labels
    logits = torch.randn((32, vocab))
    labels = torch.randint(0, vocab, (32,))
    actual = helper_func(logits, labels).cpu()
    assert (actual >= -1e-5).all(), "Cross-entropy losses must be non-negative"

    # 4. Ignore-index (-100) → loss should be 0
    logits = torch.randn((8, vocab))
    labels = torch.full((8,), -100, dtype=torch.long)
    actual = helper_func(logits, labels).cpu()
    torch.testing.assert_close(actual, torch.zeros(8), atol=1e-6, rtol=0)

    # 5. Mixed ignore-index: some -100, some valid
    labels_mixed = torch.tensor([0, -100, 3, -100, 7, -100, 1, 2])
    logits = torch.randn((8, vocab))
    expected = _ref_cross_entropy(logits, labels_mixed)
    actual = helper_func(logits, labels_mixed).cpu()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    # 6. Numerical stability – large logits
    logits_large = torch.randn((8, vocab)) * 50.0
    labels = torch.randint(0, vocab, (8,))
    expected = _ref_cross_entropy(logits_large, labels)
    actual = helper_func(logits_large, labels).cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    # 7. Determinism
    torch.manual_seed(7)
    logits = torch.randn((16, vocab))
    labels = torch.randint(0, vocab, (16,))
    out1 = helper_func(logits, labels).cpu()
    out2 = helper_func(logits, labels).cpu()
    torch.testing.assert_close(out1, out2)

def bench(kernel_fn):
    """Generate canonical benchmark inputs and invoke kernel_fn once.

    Uses (1024, 32000): 1024 tokens, vocab=32000 — representative of a
    language-model training step cross-entropy loss.
    """
    logits = torch.randn((1024, 32000))
    labels = torch.randint(0, 32000, (1024,))
    return kernel_fn(logits, labels)


# (tokens, vocab) — small / medium (matches bench()) / large
_BENCH3_SHAPES = [(512, 32000), (1024, 32000), (2048, 32000)]


def bench3(kernel_fn, idx):
    """Bench shape #idx (0..2) — 3-shape average variant of bench()."""
    tokens, vocab = _BENCH3_SHAPES[idx]
    logits = torch.randn((tokens, vocab))
    labels = torch.randint(0, vocab, (tokens,))
    return kernel_fn(logits, labels)
