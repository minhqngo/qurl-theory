"""Sanity checks for QuRL's Update-Aware Quantization (UAQ), arXiv 2602.13953, Sec. 4.3.

UAQ rescales  W -> W/s  (Q/K/V projections) and  gamma -> s*gamma  (the preceding RMSNorm gain),
with a single scalar s (s = 1.5 in the paper), before RL training starts.

Claim A (no effect on the quantized forward pass):
    With symmetric absmax quantization at per-output-channel granularity for weights and
    per-token granularity for activations (the setting stated in QuRL Sec. 5), the quantizer
    is positively homogeneous along the scaled axis, so
        Q(W/s) @ Q(sX)^T == Q(W) @ Q(X)^T     for every scalar s > 0,
    for INT8 and FP8-E4M3 alike. Hence UAQ cannot reduce the rollout quantization error.
    (A per-channel, SmoothQuant-style vector s is NOT covered and does change the error;
    it is included as a contrast.)

Claim B (UAQ is an optimizer reparameterization):
    Training (W' = W/s, gamma' = s*gamma) is trajectory-equivalent to training (W, gamma) with
        Adam: lr*s on W and lr/s on gamma   (exact as Adam eps -> 0, no clipping, wd adjusted)
        SGD : lr*s^2 on W and lr/s^2 on gamma
    so the paper's "s^2 improvement of update-to-quantization-noise ratio" is s^2 only for SGD;
    for Adam/AdamW (used by VeRL) it is s, and it equals a per-layer learning-rate change.

Run:  python uaq_equivalence.py
"""
import numpy as np

rng = np.random.default_rng(0)


def q_int8(x, axis):
    """Symmetric absmax INT8 with one scale per slice along `axis`."""
    scale = np.max(np.abs(x), axis=axis, keepdims=True) / 127.0
    return np.clip(np.round(x / scale), -127, 127) * scale


def fp8_e4m3(x):
    """Round-to-nearest-even onto the FP8 E4M3 grid (3 mantissa bits, min normal exp -6, max 448)."""
    ax = np.abs(x)
    e = np.clip(np.floor(np.log2(np.maximum(ax, 2.0 ** -30))), -6, 8)
    step = 2.0 ** (e - 3)
    return np.sign(x) * np.minimum(np.round(ax / step) * step, 448.0)


def q_fp8(x, axis):
    """Symmetric absmax FP8-E4M3 with one scale per slice along `axis`."""
    scale = np.max(np.abs(x), axis=axis, keepdims=True) / 448.0
    return fp8_e4m3(x / scale) * scale


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


# ---------------------------------------------------------------------------------------------
# Claim A: quantized forward pass is unchanged by a scalar invariant scaling
# ---------------------------------------------------------------------------------------------
print("=" * 100)
print("Claim A: Q(W/s) Q(sX) vs Q(W) Q(X)  [weights per-output-channel, activations per-token]")
print("=" * 100)
d_in, d_out, T = 1024, 512, 64
W = rng.standard_t(df=4, size=(d_out, d_in)) * 0.02  # heavy-tailed weights
x = rng.normal(size=(T, d_in))
x[:, :8] *= 30.0                                       # a few outlier channels
gamma = rng.uniform(0.5, 1.5, size=d_in)
X = gamma * rmsnorm(x)
exact = X @ W.T

for name, Q in [("INT8", q_int8), ("FP8-E4M3", q_fp8)]:
    base = Q(X, axis=1) @ Q(W, axis=1).T
    print(f"\n[{name}] relative output error without UAQ: {rel(base, exact):.6e}")
    for s in [0.7, 1.5, 2.0, 4.0]:
        Xs = (s * gamma) * rmsnorm(x)  # RMSNorm gain absorbs s
        Ws = W / s
        out = Q(Xs, axis=1) @ Q(Ws, axis=1).T
        print(f"  scalar s={s:<4}: max|out_UAQ - out_noUAQ|/max|out| = "
              f"{np.max(np.abs(out - base)) / np.max(np.abs(base)):.1e}   "
              f"relative output error with UAQ = {rel(out, exact):.6e}")
    # Contrast: a per-input-channel (SmoothQuant-like, alpha=0.5) vector does change the error.
    sv = np.sqrt(np.max(np.abs(X), axis=0) / np.max(np.abs(W), axis=0))
    out_sq = Q(X / sv, axis=1) @ Q(W * sv, axis=1).T
    print(f"  per-channel s (SmoothQuant-style, contrast): relative output error = {rel(out_sq, exact):.6e}")


# ---------------------------------------------------------------------------------------------
# Claim B: training with UAQ == training without UAQ but with per-layer learning rates
# Toy "policy": logits = (gamma * rmsnorm(x)) @ W^T, loss = -mean(A * log p(a)) (REINFORCE-like).
# ---------------------------------------------------------------------------------------------
def grads(Wp, gp, x, a, A):
    n = rmsnorm(x)
    H = gp * n
    Z = H @ Wp.T
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    P /= P.sum(axis=1, keepdims=True)
    B = x.shape[0]
    G = P.copy()
    G[np.arange(B), a] -= 1.0
    G *= A[:, None] / B                 # dL/dZ for L = -(1/B) sum_i A_i log p_i(a_i)
    gW = G.T @ H
    gg = np.sum((G @ Wp) * n, axis=0)
    return gW, gg


def train(W0, g0, lr_W, lr_g, opt, steps=300, seed=123, eps=1e-8):
    data = np.random.default_rng(seed)  # identical data stream for every run
    Wp, gp = W0.copy(), g0.copy()
    mW = vW = np.zeros_like(Wp)
    mg = vg = np.zeros_like(gp)
    b1, b2 = 0.9, 0.999
    first_step_update = None
    for t in range(1, steps + 1):
        x = data.normal(size=(32, Wp.shape[1]))
        a = data.integers(0, Wp.shape[0], size=32)
        A = data.choice([-1.0, 1.0], size=32)
        gW, gg = grads(Wp, gp, x, a, A)
        if opt == "sgd":
            dW, dg = -lr_W * gW, -lr_g * gg
        else:
            mW = b1 * mW + (1 - b1) * gW
            vW = b2 * vW + (1 - b2) * gW ** 2
            mg = b1 * mg + (1 - b1) * gg
            vg = b2 * vg + (1 - b2) * gg ** 2
            c1, c2 = 1 - b1 ** t, 1 - b2 ** t
            dW = -lr_W * (mW / c1) / (np.sqrt(vW / c2) + eps)
            dg = -lr_g * (mg / c1) / (np.sqrt(vg / c2) + eps)
        if first_step_update is None:
            first_step_update = dW.copy()
        Wp, gp = Wp + dW, gp + dg
    return Wp, gp, first_step_update


print("\n" + "=" * 100)
print("Claim B: UAQ reparameterization vs. per-layer learning-rate rescaling")
print("=" * 100)
V, d = 50, 64
W0 = rng.normal(size=(V, d)) * 0.05
g0 = rng.uniform(0.5, 1.5, size=d)
s = 1.5

for opt, lr, pw in [("adam", 1e-3, 1), ("sgd", 0.5, 2)]:
    # (1) UAQ: train W' = W0/s, gamma' = s*gamma0 with one global LR; map back W = s W', gamma = gamma'/s.
    Wu, gu, upd_u = train(W0 / s, s * g0, lr, lr, opt)
    Wu_eff, gu_eff = s * Wu, gu / s
    # (2) No UAQ, matched per-layer LRs: W gets lr*s^pw, gamma gets lr/s^pw.
    Wm, gm, _ = train(W0, g0, lr * s ** pw, lr / s ** pw, opt)
    # (3) No UAQ, plain global LR (what UAQ is usually compared against).
    Wp_, gp_, upd_p = train(W0, g0, lr, lr, opt)
    print(f"\n[{opt.upper()}]  matched LRs: W x s^{pw}, gamma / s^{pw}")
    print(f"  UAQ vs matched-LR (should be ~0):  rel diff W = {rel(Wu_eff, Wm):.2e}, gamma = {rel(gu_eff, gm):.2e}")
    print(f"  UAQ vs plain (UAQ is not a no-op): rel diff W = {rel(Wu_eff, Wp_):.2e}, gamma = {rel(gu_eff, gp_):.2e}")
    # Update-to-quantization-step ratio after the first step, in each model's own coordinates.
    step_u = np.max(np.abs(W0 / s), axis=1, keepdims=True) / 127.0
    step_p = np.max(np.abs(W0), axis=1, keepdims=True) / 127.0
    gain = np.median(np.abs(upd_u) / step_u) / np.median(np.abs(upd_p) / step_p)
    print(f"  first-step (|dW|/INT8 step) gain from UAQ = {gain:.3f}   (s = {s}, s^2 = {s * s:.2f})")
