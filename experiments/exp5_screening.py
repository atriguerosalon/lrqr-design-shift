"""E5: shifts shaped like the design application, in d = 50 with a shift spread over all coordinates.

Data: X ~ N(0, I_d) under P1. Y = w.x + sigma(x) * eps, sigma(x) = 0.1 + 0.5 * |w.x|, w = 1/sqrt(d).
Model: linear regression on source data, f(x) ~ w.x; score S = |Y - f(X)|.

S (screening): the target is the candidates a surrogate ranks in its top 10%. Candidates come from
   the same generator as calibration, so the density ratio is KNOWN: r(x) = 10 * 1{f(x) > t}.
   Weighted CP with this r is just split CP on the calibration points that pass the same filter.
O (optimizer): candidates are pushed along the surrogate gradient, x' = x + s * w, s ~ U(0, 3).
   The ratio is unknown to a practitioner; the true one is a mixture of Gaussian ratios.
"""
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from lrqr import fit_lrqr, split_cp_threshold, estimate_ratio_logistic, ess, lrqr_cv_lambda
from exp2_stress import wcp

alpha, d = 0.1, 50
N_TRAIN, N1, N3, N2, M = 5000, 2000, 2000, 5000, 5000
LAMS = [1e-3, 1e-2, 1e-1, 1.0]
LAM_GRID = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
w = np.ones(d) / np.sqrt(d)
S_GRID = np.linspace(0, 3, 61)


def response(x, rng):
    z = x @ w
    return z + (0.1 + 0.5 * np.abs(z)) * rng.normal(size=len(x))


def run(scenario, reps):
    rows = {}
    def add(k, cov, wid):
        rows.setdefault(k, [[], []]); rows[k][0].append(cov); rows[k][1].append(wid)
    for rep in range(reps):
        rng = np.random.default_rng(1000 * rep + 7)
        xt = rng.normal(size=(N_TRAIN, d)); f = LinearRegression().fit(xt, response(xt, rng))
        xc = rng.normal(size=(N1, d)); sc = np.abs(response(xc, rng) - f.predict(xc))
        xs = rng.normal(size=(N3, d))

        if scenario == "S":
            t = np.quantile(f.predict(rng.normal(size=(20000, d))), 0.9)
            def draw(n):
                out = []
                while sum(len(o) for o in out) < n:
                    x = rng.normal(size=(10 * n, d)); out.append(x[f.predict(x) > t])
                return np.concatenate(out)[:n]
            logr = lambda x: np.where(f.predict(x) > t, np.log(10.0), -np.inf)
        else:
            def draw(n):
                return rng.normal(size=(n, d)) + rng.uniform(0, 3, size=(n, 1)) * w
            def logr(x):
                z = (x @ w)[:, None]
                a = S_GRID[None, :] * z - S_GRID[None, :] ** 2 / 2
                m = a.max(1, keepdims=True)
                return (m + np.log(np.exp(a - m).mean(1, keepdims=True)))[:, 0]
        xg = draw(N2); xe = draw(M); se = np.abs(response(xe, rng) - f.predict(xe))
        sub = rng.choice(M, 1500, replace=False)

        q = split_cp_threshold(sc, alpha); add("split", np.mean(se <= q), 2 * q)
        lc = logr(xc)
        th = wcp(sc, lc, logr(xe[sub]), alpha)
        add("wcp_true_r", np.mean(se[sub] <= th), np.median(2 * th))
        rows.setdefault("_ess_true", []).append(ess(np.exp(lc - lc.max())))
        rhat = estimate_ratio_logistic(xs, xg)
        th = wcp(sc, np.log(np.maximum(rhat(xc), 1e-300)), np.log(np.maximum(rhat(xe[sub]), 1e-300)), alpha)
        add("wcp_logit_r", np.mean(se[sub] <= th), np.median(2 * th))

        # features: affine in x (already contains the linear prediction) plus its magnitude
        phi = lambda x: np.c_[np.ones(len(x)), x, np.abs(f.predict(x))]
        Pc, Ps, Pg, Pe = phi(xc), phi(xs), phi(xg), phi(xe)
        th0, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, 0.0)
        h = Pe @ th0; add("qr_lam0", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
        for l in LAMS:
            thl, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, l, theta0=th0)
            h = Pe @ thl; add(f"lrqr_{l:g}", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
        lam_cv, _ = lrqr_cv_lambda(Pc, sc, Ps, Pg, alpha, LAM_GRID, seed=rep)
        thl, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, lam_cv, theta0=th0)
        h = Pe @ thl; add("lrqr_cv", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
        rows.setdefault("_lam_cv", []).append(lam_cv)
        print(f"  rep {rep} done", flush=True)
    return rows


if __name__ == "__main__":
    from exp2_stress import show
    for sc_ in (sys.argv[1:] or ["S", "O"]):
        show(f"E5 scenario {sc_} (S = top-10% screening, O = optimizer push), d={d}", run(sc_, 10))
