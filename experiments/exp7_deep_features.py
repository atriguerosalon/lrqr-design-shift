"""E7: do features of a network trained to predict the mean lose the shift, and does LR-QR then fail?

Setting of E4 (d = 10, shift delta = 1.5 along x1, mean depends on x2 only, error size on |x1|).
A network trained only for the mean has no reason to keep x1. We take its last hidden layer as
LR-QR features, measure how much x1 information survives with a linear probe (R^2 for x1 and
|x1| on fresh data), and report coverage. E4's small one-layer net is included for comparison.
"""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from exp2_stress import sample, show, N_TRAIN, N1, N3, N2, M
from lrqr import fit_lrqr, split_cp_threshold, lrqr_cv_lambda

alpha, d, delta = 0.1, 10, 1.5
LAMS = [1e-2, 1e-1]
LAM_GRID = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
NETS = {
    "small_32": dict(hidden_layer_sizes=(32,), max_iter=500),
    "deep_128x128x64": dict(hidden_layer_sizes=(128, 128, 64), alpha=1e-4, max_iter=2000, tol=1e-6),
    "deep_wd": dict(hidden_layer_sizes=(128, 128, 64), alpha=1.0, max_iter=2000, tol=1e-6),
}


def hidden(mlp, x):
    h = x
    for W, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        h = np.maximum(0, h @ W + b)
    return h


def probe_r2(H, target):
    n = len(H) // 2
    m = RidgeCV(alphas=np.logspace(-4, 4, 17)).fit(H[:n], target[:n])
    resid = target[n:] - m.predict(H[n:])
    return 1 - resid.var() / target[n:].var()


if __name__ == "__main__":
    rows = {}
    def add(k, cov, wid):
        rows.setdefault(k, [[], []]); rows[k][0].append(cov); rows[k][1].append(wid)
    for rep in range(5):
        rng = np.random.default_rng(1000 * rep + d)
        xt, yt = sample(N_TRAIN, d, 0.0, rng)
        xc, yc = sample(N1, d, 0.0, rng)
        xs, _ = sample(N3, d, 0.0, rng)
        xg, _ = sample(N2, d, delta, rng)
        xe, ye = sample(M, d, delta, rng)
        xp, _ = sample(4000, d, 0.0, rng)                     # probe data
        for name, kw in NETS.items():
            mlp = MLPRegressor(random_state=rep, **kw).fit(xt, yt)
            sc = np.abs(yc - mlp.predict(xc)); se = np.abs(ye - mlp.predict(xe))
            if name == "small_32":
                q = split_cp_threshold(sc, alpha); add("split", np.mean(se <= q), 2 * q)
            Hp = hidden(mlp, xp)
            rows.setdefault(f"_{name}_probe_x1", []).append(probe_r2(Hp, xp[:, 0]))
            rows.setdefault(f"_{name}_probe_abs_x1", []).append(probe_r2(Hp, np.abs(xp[:, 0])))
            rows.setdefault(f"_{name}_live_units", []).append((Hp.max(0) > 0).sum())
            phi = lambda x: np.c_[np.ones(len(x)), hidden(mlp, x)]
            Pc, Ps, Pg, Pe = phi(xc), phi(xs), phi(xg), phi(xe)
            th0, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, 0.0)
            h = Pe @ th0; add(f"{name}_lam0", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
            for l in LAMS:
                thl, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, l, theta0=th0)
                h = Pe @ thl; add(f"{name}_lam{l:g}", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
            lam_cv, _ = lrqr_cv_lambda(Pc, sc, Ps, Pg, alpha, LAM_GRID, seed=rep)
            thl, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, lam_cv, theta0=th0)
            h = Pe @ thl; add(f"{name}_cv", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
        print(f"  rep {rep} done", flush=True)
    show("E7 features from networks trained for the mean, d=10, delta=1.5", rows)
