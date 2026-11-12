"""P1, Corollary 3: the whole QuRL pipeline with UAQ equals QuRL without UAQ at per-layer learning rates.

Toy QuRL loop (contextual bandit, single-token responses), float64:
  FP policy        pi(.|x) = softmax( (gamma * rmsnorm(x)) @ W^T )
  rollout policy   mu(.|x) = softmax( Q_a(gamma * rmsnorm(x)) @ Q_w(W)^T )   (INT4: per-token act., per-row weight)
  GRPO group advantages; decoupled PPO with TIS weight min(pi_old/mu, C) and the ACR upper bound
  (1+eps)/min(1, C/rho); two minibatches per rollout batch (so clipping/ACR can bind);
  AdamW (decoupled weight decay on W) with global-norm gradient clipping.

Runs (same data/sampling seeds):
  U   UAQ: train (W' = W/s, gamma' = s*gamma) with lr, Adam eps, weight decay, clip threshold.
  M   no UAQ, matched: lr_W = s*lr, lr_gamma = lr/s, eps_W = eps/s, eps_gamma = s*eps, wd_W = wd/s,
      and the clip norm computed in UAQ coordinates, i.e. on (s*g_W, g_gamma/s).
  M0  as M but with ordinary clipping (norm of (g_W, g_gamma)).
  P   no UAQ, plain hyperparameters (what UAQ is normally compared against).

Expected: U == M exactly (same sampled actions, same weights after mapping W = s*W', gamma = gamma'/s);
M0 drifts from U only through clipping; P differs.

Run:  python uaq_pipeline_equivalence.py
"""
import numpy as np

V, d, NC, G = 32, 48, 16, 4          # actions, width, contexts per batch, samples per context
B = NC * G
QMAX = 7                             # INT4 rollout quantizer: codes in [-7, 7]
EPS_CLIP, C = 0.2, 1.5               # PPO clip, TIS truncation
LR, ADAM_EPS, WD, TAU = 2e-2, 1e-8, 1e-2, 0.08
INNER_EPOCHS, N_MB = 2, 4            # minibatch reuse per rollout batch, so ratios move and ACR can bind
STEPS, S = 300, 1.5

rng = np.random.default_rng(0)
U_TRUE = rng.normal(size=(V, d))                     # "teacher" defining good actions
W_INIT = rng.normal(size=(V, d)) * 0.1
G_INIT = 1.0 + 0.1 * rng.normal(size=d)


def q_int(x, axis):
    scale = np.max(np.abs(x), axis=axis, keepdims=True) / QMAX
    scale = np.where(scale == 0, 1.0, scale)
    return np.clip(np.round(x / scale), -QMAX, QMAX) * scale


def rmsnorm(x, eps=1e-6):
    return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def policy(Wp, gp, x, quantized):
    h = gp * rmsnorm(x)
    if quantized:
        return softmax(q_int(h, axis=1) @ q_int(Wp, axis=1).T)
    return softmax(h @ Wp.T)


def adamw(p, g, m, v, t, lr, eps, wd, b1=0.9, b2=0.999):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
    return p - lr * mh / (np.sqrt(vh) + eps) - lr * wd * p, m, v


def run(variant):
    data = np.random.default_rng(7)  # identical contexts and sampling uniforms in every run
    if variant == "U":
        Wp, gp = W_INIT / S, G_INIT * S
    else:
        Wp, gp = W_INIT.copy(), G_INIT.copy()
    if variant in ("M", "M0"):
        lrW, lrg, epsW, epsg, wdW = LR * S, LR / S, ADAM_EPS / S, ADAM_EPS * S, WD / S
    else:
        lrW, lrg, epsW, epsg, wdW = LR, LR, ADAM_EPS, ADAM_EPS, WD
    mW, vW, mg, vg, t = 0 * Wp, 0 * Wp, 0 * gp, 0 * gp, 0
    actions, rewards, n_clip, n_trunc, n_acr, norms = [], [], 0, 0, 0, []
    for _ in range(STEPS):
        X = data.normal(size=(NC, d))
        X[:, :3] *= 8.0                               # outlier channels -> nontrivial quantization error
        Xb = np.repeat(X, G, axis=0)
        u = data.random(B)
        mu = policy(Wp, gp, Xb, quantized=True)       # rollout with the quantized actor
        a = np.minimum((mu.cumsum(axis=1) < u[:, None]).sum(axis=1), V - 1)
        good = np.argsort(-(Xb @ U_TRUE.T), axis=1)[:, :3]
        r = (good == a[:, None]).any(axis=1).astype(float)
        A = r - np.repeat(r.reshape(NC, G).mean(axis=1), G)
        pi_old = policy(Wp, gp, Xb, quantized=False)  # proximal policy (FP, theta_k)
        idx = np.arange(B)
        rho = pi_old[idx, a] / mu[idx, a]
        w = np.minimum(rho, C)                        # TIS weight
        upper = (1 + EPS_CLIP) / np.minimum(1.0, C / rho)   # ACR upper bound
        n_trunc += np.sum(rho > C)
        for mb in [m for _ in range(INNER_EPOCHS) for m in np.array_split(np.arange(B), N_MB)]:
            x_mb, a_mb, A_mb = Xb[mb], a[mb], A[mb]
            j = np.arange(len(a_mb))
            nrm = rmsnorm(x_mb)
            H = gp * nrm
            P = softmax(H @ Wp.T)
            R = P[j, a_mb] / pi_old[mb][j, a_mb]
            active = np.where(A_mb > 0, R < upper[mb], np.where(A_mb < 0, R > 1 - EPS_CLIP, False))
            n_acr += np.sum((A_mb > 0) & (R >= 1 + EPS_CLIP) & active)   # kept only because of ACR
            coeff = w[mb] * A_mb * active * R          # d(ell)/d(log pi(a))
            Gz = P.copy()
            Gz[j, a_mb] -= 1.0
            Gz *= coeff[:, None] / len(a_mb)           # dL/dz for L = -mean(ell)
            gW = Gz.T @ H
            gg = np.sum((Gz @ Wp) * nrm, axis=0)
            if variant == "M":
                norm = np.sqrt(np.sum((S * gW) ** 2) + np.sum((gg / S) ** 2))
            else:
                norm = np.sqrt(np.sum(gW ** 2) + np.sum(gg ** 2))
            kappa = min(1.0, TAU / (norm + 1e-300))
            n_clip += kappa < 1.0
            norms.append(norm)
            t += 1
            Wp, mW, vW = adamw(Wp, kappa * gW, mW, vW, t, lrW, epsW, wdW)
            gp, mg, vg = adamw(gp, kappa * gg, mg, vg, t, lrg, epsg, 0.0)
        actions.append(a)
        rewards.append(r.mean())
    if variant == "U":                                # map back to original coordinates
        Wp, gp = S * Wp, gp / S
    return dict(W=Wp, g=gp, actions=np.array(actions), reward=np.array(rewards),
                clip_frac=n_clip / (INNER_EPOCHS * N_MB * STEPS), trunc=n_trunc / (B * STEPS), acr=n_acr,
                nonzero=np.mean(np.array(norms) > 0), med_norm=np.median([x for x in norms if x > 0]))


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


runs = {k: run(k) for k in ["U", "M", "M0", "P"]}
u = runs["U"]
print(f"activity in run U: {100 * u['nonzero']:.0f}% of minibatch updates have a nonzero gradient (the rest have\n"
      f"  all-equal group rewards once the toy policy turns deterministic); clipping active on "
      f"{100 * u['clip_frac']:.0f}% of updates\n  (median nonzero grad norm {u['med_norm']:.2e}, threshold {TAU}); "
      f"TIS truncation on {100 * u['trunc']:.1f}% of samples;\n  ACR kept {u['acr']} positive-sample gradient terms "
      f"that plain TIS clipping would have dropped\n")
print(f"{'run':>4} {'same actions':>13} {'rel diff W vs U':>16} {'rel diff gamma vs U':>20} {'reward (last 100)':>18}")
for k, rr in runs.items():
    same = np.mean(rr["actions"] == u["actions"])
    print(f"{k:>4} {100 * same:>12.2f}% {rel(rr['W'], u['W']):>16.2e} {rel(rr['g'], u['g']):>20.2e} "
          f"{rr['reward'][-100:].mean():>18.3f}")
print("\nU == M: UAQ is exactly per-layer learning rates (plus eps/wd/clipping in matching coordinates).")
