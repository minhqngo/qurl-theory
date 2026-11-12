"""Does round-to-nearest (RTN) quantization of full-precision master weights 'freeze' the rollout
model when updates are far below the quantization step?  (QuRL Sec. 4.3 / Appendix A claim.)

Setting: master weights W (rows x n) are re-quantized every RL step with per-row absmax INT8
(dynamic weight quantization, as when weights are re-loaded into the inference engine).
After a cumulative full-precision update D, compare the change of the quantized layer output
    dy_Q  = (Q(W + D) - Q(W)) x       against the full-precision change   dy_FP = D x.

Uniform-phase prediction (weights' positions inside quantization cells ~ uniform, |d| <= step):
    P(code flips) ~= |d|/step,   E[Q(w+d) - Q(w)] ~= d   (unbiased in aggregate),
    Var[Q(w+d) - Q(w)] ~= step*|d| - d^2.
So the quantized model tracks the update in expectation; the issue is variance (SNR), which
depends on how coherent the update is with the inputs -- not on ||D|| / ||Q(W) - W||.

Run:  python rtn_tracking.py
"""
import numpy as np

rng = np.random.default_rng(1)
rows, n = 512, 2048


def absmax_step(W):
    return np.max(np.abs(W), axis=1, keepdims=True) / 127.0


def codes(W, step):
    return np.clip(np.round(W / step), -127, 127)


W = rng.normal(size=(rows, n)) * 0.02
step0 = absmax_step(W)
c0 = codes(W, step0)
QW = c0 * step0
x = rng.normal(size=n)  # probe input
rel_quant_err = np.linalg.norm(QW - W) / np.linalg.norm(W)
print(f"relative weight quantization error ||Q(W)-W||/||W|| = {rel_quant_err:.3e}")
print("scale = 'frozen': per-row step fixed at its initial value; 'dynamic': absmax recomputed after\n"
      "the update (what re-quantizing every RL step does; a moving absmax also rescales the grid).\n")
print(f"{'|d|/step':>9} {'update':>8} {'scale':>8} {'||D||/||W||':>12} {'code flips':>11} {'pred':>7} "
      f"{'slope':>7} {'corr':>6} {'resid std':>10} {'pred std':>9}")

for r in [1e-3, 1e-2, 1e-1, 3e-1]:
    for kind in ["aligned", "random"]:
        if kind == "aligned":   # Adam-like sign update aligned with the probe input (coherent)
            dirn = np.sign(x)[None, :] * rng.choice([-1.0, 1.0], size=(rows, 1))
        else:                   # random-sign update, incoherent with the probe input
            dirn = rng.choice([-1.0, 1.0], size=(rows, n))
        D = r * step0 * dirn
        for scale in ["frozen", "dynamic"]:
            step1 = step0 if scale == "frozen" else absmax_step(W + D)
            c1 = codes(W + D, step1)
            flips = np.mean(c1 != c0)
            dy_fp = D @ x
            dy_q = (c1 * step1 - QW) @ x
            slope = np.dot(dy_q, dy_fp) / np.dot(dy_fp, dy_fp)
            corr = np.corrcoef(dy_q, dy_fp)[0, 1]
            resid = np.std(dy_q - dy_fp)
            pred = np.sqrt(np.mean((step0[:, 0] ** 2) * r * (1 - r) * np.sum(x ** 2)))
            print(f"{r:>9.0e} {kind:>8} {scale:>8} {np.linalg.norm(D) / np.linalg.norm(W):>12.2e} "
                  f"{flips:>11.4f} {r:>7.4f} {slope:>7.3f} {corr:>6.3f} {resid:>10.2e} {pred:>9.2e}")

print("\nReading: with a frozen grid, code flips ~= |d|/step and the slope ~= 1 at every scale, so RTN\n"
      "does not 'freeze' the rollout model in expectation; it adds tracking noise with variance\n"
      "~ step*|d| per weight. Gradient-aligned updates are tracked with SNR ~ sqrt(n*|d|/step) even at\n"
      "|d|/step = 1e-3, while ||D||/||W|| is far below ||Q(W)-W||/||W|| (the paper's Appendix-A metric).\n"
      "Random-direction updates are tracked with SNR ~ sqrt(|d|/step) only (the slope estimate is noisy\n"
      "there because the signal is tiny). Frozen vs dynamic absmax makes little difference for these\n"
      "Gaussian rows; outlier-dominated rows were not tested.")
