"""E1: replicate the authors' rebuttal simulation (OpenReview reply to Reviewer KYG7, 7 Aug 2025).

Setting (as described there): d = 10, P1,X = N(-1/sqrt(d) * 1, I/d), P2,X = N(+1/sqrt(d) * 1, I/d),
Y | X = x ~ N(x_1, 1/d). Linear regression fit on n_train = 10000 source points; score
S = |Y - f(X)|; n1 = 5000 labeled calibration, n3 = 5000 unlabeled source, n2 = 10000 unlabeled
target; affine hypothesis class; alpha = 0.1; lambda in {1e-6, ..., 1e-2}; 20 repetitions.
Their reported coverages: 0.864, 0.860, 0.867, 0.943, 1.000.

Note: the noise is homoscedastic and the model well specified, so the score distribution does
not depend on x. Split CP should then already be valid under this covariate shift.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from lrqr import fit_lrqr, split_cp_threshold, estimate_ratio_logistic

d, alpha, reps = 10, 0.1, 20
n_train, n1, n3, n2, m = 10000, 5000, 5000, 10000, 10000
mu = np.ones(d) / np.sqrt(d)
sd = 1 / np.sqrt(d)
lams = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

def sample(n, sign, rng):
    x = rng.normal(sign * mu, sd, size=(n, d))
    y = x[:, 0] + rng.normal(0, sd, size=n)
    return x, y

def weighted_cp_logspace(s_cal, logw_cal, logw_test, alpha):
    order = np.argsort(s_cal)
    s_sorted, lw = s_cal[order], logw_cal[order]
    out = np.empty(len(logw_test))
    for j, lt in enumerate(logw_test):
        M = max(lw.max(), lt)
        w = np.exp(lw - M); wt = np.exp(lt - M)
        p = np.cumsum(w) / (w.sum() + wt)
        idx = np.searchsorted(p, 1 - alpha)
        out[j] = np.inf if idx >= len(s_sorted) else s_sorted[idx]
    return out

res = {k: [] for k in ["split", "wcp_true", "wcp_logit", "qr0"] + [f"lrqr_{l:g}" for l in lams]}
width = {k: [] for k in res}
for rep in range(reps):
    rng = np.random.default_rng(rep)
    xt, yt = sample(n_train, -1, rng)
    f = LinearRegression().fit(xt, yt)
    xc, yc = sample(n1, -1, rng); sc = np.abs(yc - f.predict(xc))
    xs, _ = sample(n3, -1, rng)
    xg, _ = sample(n2, +1, rng)
    xe, ye = sample(m, +1, rng); se = np.abs(ye - f.predict(xe))
    phi = lambda x: np.c_[np.ones(len(x)), x]

    q = split_cp_threshold(sc, alpha)
    res["split"].append(np.mean(se <= q)); width["split"].append(2 * q)

    logr = lambda x: 2 * np.sqrt(d) * x.sum(1)          # true log likelihood ratio
    sub = rng.choice(m, 2000, replace=False)             # evaluate weighted CP on a subsample (cost)
    th = weighted_cp_logspace(sc, logr(xc), logr(xe[sub]), alpha)
    res["wcp_true"].append(np.mean(se[sub] <= th)); width["wcp_true"].append(np.median(2 * th))

    rhat = estimate_ratio_logistic(xs, xg)
    lr_c = np.log(np.maximum(rhat(xc), 1e-300)); lr_e = np.log(np.maximum(rhat(xe[sub]), 1e-300))
    th = weighted_cp_logspace(sc, lr_c, lr_e, alpha)
    res["wcp_logit"].append(np.mean(se[sub] <= th)); width["wcp_logit"].append(np.median(2 * th))

    th0, _ = fit_lrqr(phi(xc), sc, phi(xs), phi(xg), alpha, 0.0)
    h = phi(xe) @ th0
    res["qr0"].append(np.mean(se <= h)); width["qr0"].append(np.mean(2 * np.maximum(h, 0)))
    for l in lams:
        thl, _ = fit_lrqr(phi(xc), sc, phi(xs), phi(xg), alpha, l, theta0=th0)
        h = phi(xe) @ thl
        res[f"lrqr_{l:g}"].append(np.mean(se <= h)); width[f"lrqr_{l:g}"].append(np.mean(2 * np.maximum(h, 0)))

print(f"E1 authors' toy, d={d}, alpha={alpha}, {reps} reps; E1[r^2] = exp(4d) = {np.exp(4*d):.2e}")
print(f"{'method':14s} {'coverage':>18s} {'width':>10s}")
for k in res:
    c = np.array(res[k]); w = np.array(width[k])
    print(f"{k:14s} {c.mean():.3f} +- {c.std():.3f}    {np.median(w):10.3f}")
