"""P1, Section 5: how a re-quantized rollout model tracks drifting full-precision master weights.

Checks, on n coordinates with a smooth weight distribution (uniform phase holds):
  Prop. 6(i)   the RTN error e_k = Q(theta_k) - theta_k stays U[-D/2, D/2]  (variance D^2/12 at every k)
  Prop. 6(ii)  Corr(e(theta + d), e(theta)) = 1 - 6a(1-a),  a = frac(|d|/D)
  Prop. 7      per-step error variance, lag-1 autocorrelation, and growth of the partial sums
               sum_{k<K} e_k for four rollout-quantization schemes:
                 RTN       round-to-nearest of theta_k (fixed grid)
                 SR        stochastic rounding with fresh randomness every step
                 EF        error feedback / sigma-delta:  q_k = Q(theta_k + r_{k-1}),  r_k = theta_k + r_{k-1} - q_k
                 DELTA     Q(theta_0) + Q_fine(theta_k - theta_0) with a 64x finer grid for the delta
             under Adam-like per-step updates of size eta = 0.01*D (random-walk and coherent drift).
The partial sums matter because a first-order mismatch bias b_k ~ B e_k accumulates as B * sum_k e_k.

Run:  python tracking_schemes.py
"""
import numpy as np

rng = np.random.default_rng(3)
n, D = 50_000, 1.0
theta0 = rng.normal(size=n) * 10.0  # smooth on the grid scale -> phases ~ uniform


def rtn(x, step=D):
    return np.round(x / step) * step


# ---- Prop. 6(ii): persistence of the RTN error -------------------------------------------------
e0 = rtn(theta0) - theta0
print("Prop. 6(ii): Corr(e(theta0 + d), e(theta0)) vs 1 - 6a(1-a)")
for a in [0.01, 0.05, 1 / 12, 0.25, 0.5, 0.75, 1.3]:
    d = a * D * rng.choice([-1.0, 1.0], size=n)
    e1 = rtn(theta0 + d) - (theta0 + d)
    fa = a % 1.0
    print(f"  |d|/D = {a:5.3f}: empirical {np.corrcoef(e0, e1)[0, 1]:+.4f}   predicted {1 - 6 * fa * (1 - fa):+.4f}")

# ---- Prop. 6(i) + Prop. 7: schemes over K steps -------------------------------------------------
K, eta = 2000, 0.01 * D
checkpoints = [10, 100, 1000, 2000]
for drift in ["random-walk", "coherent"]:
    theta = theta0.copy()
    sign0 = rng.choice([-1.0, 1.0], size=n)
    r_ef = np.zeros(n)
    base = rtn(theta0)
    names = ["RTN", "SR", "EF", "DELTA"]
    S = {s: np.zeros(n) for s in names}
    sq = {s: 0.0 for s in names}
    lag = {s: 0.0 for s in names}
    prev = {s: None for s in names}
    ps_rms = {s: {} for s in names}
    var_rtn_first = var_rtn_last = None
    for k in range(1, K + 1):
        e = {
            "RTN": rtn(theta) - theta,
            "SR": D * np.floor(theta / D + rng.random(n)) - theta,
        }
        q_ef = rtn(theta + r_ef)
        e["EF"] = q_ef - theta
        r_ef = theta + r_ef - q_ef
        delta = theta - theta0
        e["DELTA"] = base + rtn(delta, D / 64) - theta
        for s in names:
            S[s] += e[s]
            sq[s] += np.mean(e[s] ** 2)
            if prev[s] is not None:
                lag[s] += np.mean(e[s] * prev[s]) / np.sqrt(np.mean(e[s] ** 2) * np.mean(prev[s] ** 2))
            prev[s] = e[s]
            if k in checkpoints:
                ps_rms[s][k] = np.sqrt(np.mean(S[s] ** 2)) / D
        if k == 1:
            var_rtn_first = np.mean(e["RTN"] ** 2)
        if k == K:
            var_rtn_last = np.mean(e["RTN"] ** 2)
        theta = theta + (eta * rng.choice([-1.0, 1.0], size=n) if drift == "random-walk" else eta * sign0)
    print(f"\nDrift = {drift}, per-step update {eta / D:.2f}*D, K = {K} steps")
    print(f"  Prop. 6(i): RTN error variance *12/D^2 at k=1: {12 * var_rtn_first:.4f}, at k=K: {12 * var_rtn_last:.4f}")
    print(f"  {'scheme':<6} {'per-step var*12/D^2':>20} {'lag-1 corr':>11} " +
          " ".join(f"{'rms sum K=' + str(c):>14}" for c in checkpoints))
    for s in names:
        print(f"  {s:<6} {12 * sq[s] / K:>20.3f} {lag[s] / (K - 1):>11.3f} " +
              " ".join(f"{ps_rms[s][c]:>14.2f}" for c in checkpoints))

print("\nReading: RTN keeps the D^2/12 error forever (stationary) but it is persistent (lag-1 corr ~1),\n"
      "so its partial sums grow ~linearly until the drift decorrelates it; DELTA captures every update\n"
      "yet has the most persistent error; SR is white (sums grow ~sqrt(K)); EF has bounded partial sums.")
