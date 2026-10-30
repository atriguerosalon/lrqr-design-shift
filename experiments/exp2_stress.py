"""E2-E4: stress tests where the covariate shift actually matters.

Data: X ~ N(0, I_d) under P1 (calibration), X ~ N(delta * e1, I_d) under P2 (target).
Y = x2 + sigma(x) * eps,  sigma(x) = 0.1 + 0.5 * |x1|  (errors grow along the shift direction).
Model: linear regression on 5000 source points (well specified for the mean), score S = |Y - f(X)|.
So split CP must under-cover under the shift. True log-ratio: delta * x1 - delta^2 / 2;
E1[r^2] = exp(delta^2).

E2: shift strength delta, d = 10, affine features.
E3: dimension d, delta = 1.5, affine features (WCP with logistic-ratio vs LR-QR).
E4: features that cannot see the shift (drop x1, or use features learned for predicting the mean).
"""
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from lrqr import fit_lrqr, split_cp_threshold, estimate_ratio_logistic, ess, lrqr_cv_lambda

alpha = 0.1
N_TRAIN, N1, N3, N2, M = 5000, 2000, 2000, 5000, 5000
LAMS = [1e-3, 1e-2, 1e-1, 1.0]
LAM_GRID = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def sample(n, d, delta, rng):
    x = rng.normal(size=(n, d)); x[:, 0] += delta
    y = x[:, 1] + (0.1 + 0.5 * np.abs(x[:, 0])) * rng.normal(size=n)
    return x, y


def wcp(s_cal, logw_cal, logw_test, alpha):
    order = np.argsort(s_cal); s_sorted, lw = s_cal[order], logw_cal[order]
    out = np.empty(len(logw_test))
    for j, lt in enumerate(logw_test):
        Mx = max(lw.max(), lt); w = np.exp(lw - Mx); wt = np.exp(lt - Mx)
        idx = np.searchsorted(np.cumsum(w) / (w.sum() + wt), 1 - alpha)
        out[j] = np.inf if idx >= len(s_sorted) else s_sorted[idx]
    return out


def run(d, delta, reps, feats="affine", with_cv=False, bounded=False):
    T = (lambda s: s / (1.0 + s)) if bounded else (lambda s: s)
    rows = {}
    def add(k, cov, wid):
        rows.setdefault(k, [[], []]); rows[k][0].append(cov); rows[k][1].append(wid)
    for rep in range(reps):
        rng = np.random.default_rng(1000 * rep + d)
        xt, yt = sample(N_TRAIN, d, 0.0, rng)
        f = LinearRegression().fit(xt, yt)
        xc, yc = sample(N1, d, 0.0, rng); sc = T(np.abs(yc - f.predict(xc)))
        xs, _ = sample(N3, d, 0.0, rng)
        xg, _ = sample(N2, d, delta, rng)
        xe, ye = sample(M, d, delta, rng); se = T(np.abs(ye - f.predict(xe)))
        sub = rng.choice(M, 1500, replace=False)

        if feats == "affine":
            phi = lambda x: np.c_[np.ones(len(x)), x]
        elif feats == "rich":             # affine + squares + absolute values of every coordinate
            phi = lambda x: np.c_[np.ones(len(x)), x, x ** 2, np.abs(x)]
        elif feats == "blind":            # features that do not see the shifted coordinate x1
            phi = lambda x: np.c_[np.ones(len(x)), x[:, 1:]]
        elif feats == "learned":          # hidden features of an MLP trained to predict the mean
            mlp = MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=rep).fit(xt, yt)
            def phi(x):
                hdn = np.maximum(0, x @ mlp.coefs_[0] + mlp.intercepts_[0])
                return np.c_[np.ones(len(x)), hdn]
        elif feats == "learned_noise":    # MLP trained to predict |residual|: features do see x1
            res_t = np.abs(yt - f.predict(xt))
            mlp = MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=rep).fit(xt, res_t)
            def phi(x):
                hdn = np.maximum(0, x @ mlp.coefs_[0] + mlp.intercepts_[0])
                return np.c_[np.ones(len(x)), hdn]

        q = split_cp_threshold(sc, alpha); add("split", np.mean(se <= q), 2 * q)
        logr = lambda x: delta * x[:, 0] - delta ** 2 / 2
        th = wcp(sc, logr(xc), logr(xe[sub]), alpha)
        add("wcp_true_r", np.mean(se[sub] <= th), np.median(2 * th))
        rhat = estimate_ratio_logistic(xs, xg)
        th = wcp(sc, np.log(np.maximum(rhat(xc), 1e-300)), np.log(np.maximum(rhat(xe[sub]), 1e-300)), alpha)
        add("wcp_logit_r", np.mean(se[sub] <= th), np.median(2 * th))
        rows.setdefault("_ess_true", []).append(ess(np.exp(logr(xc) - logr(xc).max())))

        Pc, Ps, Pg, Pe = phi(xc), phi(xs), phi(xg), phi(xe)
        th0, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, 0.0)
        h = Pe @ th0; add("qr_lam0", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
        for l in LAMS:
            thl, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, l, theta0=th0)
            h = Pe @ thl; add(f"lrqr_{l:g}", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
        if with_cv:
            lam_cv, _ = lrqr_cv_lambda(Pc, sc, Ps, Pg, alpha, LAM_GRID, seed=rep)
            thl, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, lam_cv, theta0=th0)
            h = Pe @ thl; add("lrqr_cv", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
            rows.setdefault("_lam_cv", []).append(lam_cv)
    return rows


def show(title, rows):
    print(f"\n== {title}")
    for k, v in rows.items():
        if k.startswith("_"):
            print(f"   {k:12s} median {np.median(v):.3g}")
            continue
        c, w = np.array(v[0]), np.array(v[1])
        wmed = np.median(w)
        print(f"   {k:12s} cov {c.mean():.3f} +- {c.std():.3f}   width {wmed:8.3f}"
              + ("  (inf in %d%%)" % (100 * np.mean(np.isinf(w))) if np.isinf(w).any() else ""))


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    REPS = 10
    if part in ("E2", "all"):
        for delta in [0.5, 1.0, 1.5, 2.0, 3.0]:
            show(f"E2 shift strength delta={delta} (E1[r^2]={np.exp(delta**2):.3g}), d=10, affine",
                 run(10, delta, REPS, "affine", with_cv=True))
    if part in ("E2rich", "all"):
        for delta in [0.5, 1.0, 1.5, 2.0, 3.0]:
            show(f"E2 shift strength delta={delta} (E1[r^2]={np.exp(delta**2):.3g}), d=10, rich",
                 run(10, delta, REPS, "rich", with_cv=True))
    if part == "E2rich_bounded":
        for delta in [1.0, 2.0, 3.0]:
            show(f"E2 bounded score S/(1+S), delta={delta}, d=10, rich",
                 run(10, delta, REPS, "rich", with_cv=True, bounded=True))
    if part in ("E3", "all"):
        for d in [5, 20, 50, 100, 200]:
            show(f"E3 dimension d={d}, delta=1.5, rich features", run(d, 1.5, 5, "rich"))
    if part in ("E4", "all"):
        for feats in ["affine", "blind", "learned", "learned_noise"]:
            show(f"E4 features={feats}, d=10, delta=1.5", run(10, 1.5, REPS, feats))
