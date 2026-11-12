"""QuRL Fig. 4(b) plots the average of (pi_behav - pi_prox) over tokens SAMPLED from pi_behav.
That statistic is positive even when the two policies are exchangeable perturbations of the same
logits, because
    E_{a ~ mu}[mu(a) - pi(a)] = 0.5*||mu - pi||_2^2 + 0.5*(||mu||_2^2 - ||pi||_2^2),
and the second term has mean zero under exchangeability. So a positive curve alone does not show
that quantization systematically sharpens the rollout policy; it is mostly a sampling artifact
that scales with ||mu - pi||^2.

Run:  python selection_bias_metric.py
"""
import numpy as np

rng = np.random.default_rng(2)
V, contexts = 2000, 4000
vals, sym = [], []
for _ in range(contexts):
    z = rng.normal(size=V) * 3.0                    # base logits
    mu = np.exp(z + rng.normal(size=V) * 0.3)       # "quantized" policy: logit noise
    pi = np.exp(z + rng.normal(size=V) * 0.3)       # "full-precision" policy: same noise law
    mu /= mu.sum()
    pi /= pi.sum()
    vals.append(np.dot(mu, mu - pi))                # exact E_{a~mu}[mu(a) - pi(a)]
    sym.append(0.5 * np.sum((mu - pi) ** 2))
print(f"mean E_(a~mu)[mu(a)-pi(a)]   = {np.mean(vals):.4e}  (+/- {np.std(vals) / np.sqrt(contexts):.1e})")
print(f"mean 0.5*||mu-pi||^2         = {np.mean(sym):.4e}")
print("Positive although mu and pi are exchangeable: the metric is biased by sampling from mu.")
