"""P1, Theorem 1 boundaries: when does a scalar invariant scaling (W/s, s*X) change quantization?

Theorem 1 says: never, if every scale level is an absmax-derived (dynamic) real number and s is constant
inside each quantization group. This script probes the cases outside that condition:
  (a) INT8 per-channel/per-token, dynamic scales (Theorem 1 holds)          -> no change
  (b) NVFP4-like: E2M1 elements, E4M3-rounded block scales under a dynamic
      per-tensor FP32 scale (two-level, all absmax-derived)                   -> no change
  (c) MXFP4: E2M1 elements, power-of-two (E8M0) block scales               -> changes unless s = 2^k
  (d) static activation scale calibrated before UAQ and not recalibrated    -> changes (a calibration change)
  (e) direct FP8 cast with no scale factor                                  -> changes unless s = 2^k
  (f) INT8 of BF16-stored weights: bf16(W/s) vs bf16(W)                     -> a few random code changes
For each case the relative output error of the quantized layer is printed for s in {1, 1.5, 2, 3}.
Changes in (c)-(f) are rounding-pattern changes, not systematic error reductions.

Run:  python uaq_edge_cases.py
"""
import numpy as np

rng = np.random.default_rng(5)
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def fp8_e4m3(x):
    ax = np.abs(x)
    e = np.clip(np.floor(np.log2(np.maximum(ax, 2.0 ** -30))), -6, 8)
    step = 2.0 ** (e - 3)
    return np.sign(x) * np.minimum(np.round(ax / step) * step, 448.0)


def e2m1(x):
    ax = np.minimum(np.abs(x), 6.0)
    return np.sign(x) * E2M1[np.abs(ax[..., None] - E2M1).argmin(axis=-1)]


def bf16(x):
    m, e = np.frexp(x)
    return np.ldexp(np.round(m * 256.0) / 256.0, e)


def q_int8(x, axis, scale=None):
    if scale is None:
        scale = np.max(np.abs(x), axis=axis, keepdims=True) / 127.0
    return np.clip(np.round(x / scale), -127, 127) * scale


def q_nvfp4(x, block=16):
    g = np.max(np.abs(x)) / (6.0 * 448.0)                      # dynamic per-tensor FP32 scale
    xb = x.reshape(-1, block)
    sb = fp8_e4m3(np.max(np.abs(xb), axis=1, keepdims=True) / (6.0 * g)) * g   # E4M3 block scale
    sb = np.where(sb == 0, 1.0, sb)
    return (e2m1(xb / sb) * sb).reshape(x.shape)


def q_mxfp4(x, block=32):
    xb = x.reshape(-1, block)
    amax = np.max(np.abs(xb), axis=1, keepdims=True)
    sb = 2.0 ** (np.floor(np.log2(np.maximum(amax, 1e-30))) - 2)   # E8M0 shared exponent, emax(E2M1) = 2
    return (e2m1(xb / sb) * sb).reshape(x.shape)


d_in, d_out, T = 1024, 256, 64
W = rng.standard_t(df=4, size=(d_out, d_in)) * 0.02
x = rng.normal(size=(T, d_in))
x[:, :8] *= 30.0
gamma = rng.uniform(0.5, 1.5, size=d_in)
X = gamma * x / np.sqrt(np.mean(x ** 2, axis=1, keepdims=True))
exact = X @ W.T
calib_scale = np.max(np.abs(X)) / 127.0            # static per-tensor activation scale, calibrated pre-UAQ

cases = {
    "(a) INT8 dynamic (Theorem 1)": lambda Xs, Ws: q_int8(Xs, 1) @ q_int8(Ws, 1).T,
    "(b) NVFP4-like two-level":     lambda Xs, Ws: q_nvfp4(Xs) @ q_nvfp4(Ws).T,
    "(c) MXFP4 pow-2 scales":       lambda Xs, Ws: q_mxfp4(Xs) @ q_mxfp4(Ws).T,
    "(d) INT8 static act. scale":   lambda Xs, Ws: q_int8(Xs, None, calib_scale) @ q_int8(Ws, 1).T,
    "(e) FP8 cast, no scale":       lambda Xs, Ws: fp8_e4m3(Xs) @ fp8_e4m3(Ws).T,
    "(f) INT8 of bf16 weights":     lambda Xs, Ws: q_int8(Xs, 1) @ q_int8(bf16(Ws), 1).T,
}

print(f"{'case':<32} " + " ".join(f"{'s=' + str(s):>22}" for s in [1.0, 1.5, 2.0, 3.0]))
print(f"{'':<32} " + " ".join(f"{'rel.err (max diff)':>22}" for _ in range(4)))
for name, f in cases.items():
    base = f(X, W)
    cells = []
    for s in [1.0, 1.5, 2.0, 3.0]:
        out = f(s * X, W / s)
        err = np.linalg.norm(out - exact) / np.linalg.norm(exact)
        diff = np.max(np.abs(out - base)) / np.max(np.abs(base))
        cells.append(f"{err:.4e} ({diff:.0e})")
    print(f"{name:<32} " + " ".join(f"{c:>22}" for c in cells))

# (f) in detail: how many INT8 weight codes change because bf16 re-rounds W/s?
for s in [1.5, 3.0]:
    W0, W1 = bf16(W), bf16(W / s)
    c0 = np.round(W0 / (np.max(np.abs(W0), axis=1, keepdims=True) / 127.0))
    c1 = np.round(W1 / (np.max(np.abs(W1), axis=1, keepdims=True) / 127.0))
    print(f"(f) s={s}: fraction of INT8 weight codes changed by bf16 re-rounding = {np.mean(c0 != c1):.2e}")
