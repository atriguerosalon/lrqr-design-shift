"""E8: a usable warning check for LR-QR's silent failure, and a feature fix.

Check (needs only labeled source data): can the features predict the calibration scores?
  skill = 1 - pinball(quantile regression on phi) / pinball(constant quantile), 2-fold on calibration.
  skill ~ 0 means the features cannot express where errors are large, so LR-QR cannot correct for it.
Fix: fit a model g(x) of the score on an extra labeled source split and append g(x) to the features.
  Two choices of g: gradient boosting (extrapolates flat) and an MLP (extrapolates roughly linearly).
  The extra split costs labeled data; in practice it would come out of the calibration budget.

Cases (5 reps each):
  E4_blind   d=10, shift 1.5 along x1, features without x1             (silent failure in E4)
  E4_affine  same data, affine features                                (partial failure in E2)
  E7_deepwd  same data, features of a weight-decayed deep net          (failure in E7)
  E5O_affine d=50, optimizer push, affine features
  E5O_rich   d=50, optimizer push, [1, x, |f(x)|]                       (control: worked in E5)
"""
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from lrqr import fit_lrqr, split_cp_threshold, lrqr_cv_lambda
import exp2_stress as e2
import exp5_screening as e5
from exp7_deep_features import hidden

alpha = 0.1
N_AUX = 2000
LAMS = [1e-2, 1e-1]
LAM_GRID = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def pinball(s, q, tau=1 - alpha):
    u = s - q
    return np.mean(np.maximum(tau * u, (tau - 1) * u))


def score_skill(P, s):
    n = len(s) // 2
    out = []
    for a, b in [(slice(0, n), slice(n, None)), (slice(n, None), slice(0, n))]:
        th, _ = fit_lrqr(P[a], s[a], P[a], P[a], alpha, 0.0)
        out.append(1 - pinball(s[b], P[b] @ th) / pinball(s[b], np.quantile(s[a], 1 - alpha)))
    return np.mean(out)


def make_case(case, rep):
    """Returns x arrays, scores, base feature map and the input of the score model."""
    if case.startswith("E4") or case.startswith("E7"):
        d, delta = 10, 1.5
        rng = np.random.default_rng(1000 * rep + d)
        xt, yt = e2.sample(e2.N_TRAIN, d, 0.0, rng)
        if case == "E7_deepwd":
            f = MLPRegressor(hidden_layer_sizes=(128, 128, 64), alpha=1.0, max_iter=2000, tol=1e-6,
                             random_state=rep).fit(xt, yt)
        else:
            f = LinearRegression().fit(xt, yt)
        xc, yc = e2.sample(e2.N1, d, 0.0, rng)
        xa, ya = e2.sample(N_AUX, d, 0.0, rng)
        xs, _ = e2.sample(e2.N3, d, 0.0, rng)
        xg, _ = e2.sample(e2.N2, d, delta, rng)
        xe, ye = e2.sample(e2.M, d, delta, rng)
        phi = {"E4_blind": lambda x: np.c_[np.ones(len(x)), x[:, 1:]],
               "E4_affine": lambda x: np.c_[np.ones(len(x)), x],
               "E7_deepwd": lambda x: np.c_[np.ones(len(x)), hidden(f, x)]}[case]
    else:
        d = e5.d
        rng = np.random.default_rng(1000 * rep + 7)
        xt = rng.normal(size=(e5.N_TRAIN, d)); f = LinearRegression().fit(xt, e5.response(xt, rng))
        xc = rng.normal(size=(e5.N1, d)); yc = e5.response(xc, rng)
        xa = rng.normal(size=(N_AUX, d)); ya = e5.response(xa, rng)
        xs = rng.normal(size=(e5.N3, d))
        draw = lambda n: rng.normal(size=(n, d)) + rng.uniform(0, 3, size=(n, 1)) * e5.w
        xg = draw(e5.N2); xe = draw(e5.M); ye = e5.response(xe, rng)
        phi = {"E5O_affine": lambda x: np.c_[np.ones(len(x)), x],
               "E5O_rich": lambda x: np.c_[np.ones(len(x)), x, np.abs(f.predict(x))]}[case]
    sc, se, sa = (np.abs(y - f.predict(x)) for x, y in [(xc, yc), (xe, ye), (xa, ya)])
    return dict(xc=xc, xs=xs, xg=xg, xe=xe, xa=xa, sc=sc, se=se, sa=sa, phi=phi)


def run(case, reps=5):
    rows = {}
    def add(k, cov, wid):
        rows.setdefault(k, [[], []]); rows[k][0].append(cov); rows[k][1].append(wid)
    for rep in range(reps):
        D = make_case(case, rep)
        q = split_cp_threshold(D["sc"], alpha); add("split", np.mean(D["se"] <= q), 2 * q)
        g_models = {
            "gbm": HistGradientBoostingRegressor(random_state=rep).fit(D["xa"], D["sa"]),
            "mlp": MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=1000,
                                early_stopping=True, random_state=rep).fit(D["xa"], D["sa"]),
        }
        maps = {"base": D["phi"]}
        for name, g in g_models.items():
            maps[f"+g_{name}"] = (lambda g: lambda x: np.c_[D["phi"](x), g.predict(x)])(g)
        for name, phi in maps.items():
            Pc, Ps, Pg, Pe = phi(D["xc"]), phi(D["xs"]), phi(D["xg"]), phi(D["xe"])
            rows.setdefault(f"_skill_{name}", []).append(score_skill(Pc, D["sc"]))
            th0, _ = fit_lrqr(Pc, D["sc"], Ps, Pg, alpha, 0.0)
            h = Pe @ th0; add(f"{name}_lam0", np.mean(D["se"] <= h), np.mean(2 * np.maximum(h, 0)))
            for l in LAMS:
                thl, _ = fit_lrqr(Pc, D["sc"], Ps, Pg, alpha, l, theta0=th0)
                h = Pe @ thl; add(f"{name}_lam{l:g}", np.mean(D["se"] <= h), np.mean(2 * np.maximum(h, 0)))
            lam_cv, _ = lrqr_cv_lambda(Pc, D["sc"], Ps, Pg, alpha, LAM_GRID, seed=rep)
            thl, _ = fit_lrqr(Pc, D["sc"], Ps, Pg, alpha, lam_cv, theta0=th0)
            h = Pe @ thl; add(f"{name}_cv", np.mean(D["se"] <= h), np.mean(2 * np.maximum(h, 0)))
        print(f"  {case} rep {rep} done", flush=True)
    return rows


if __name__ == "__main__":
    for case in (sys.argv[1:] or ["E4_blind", "E4_affine", "E5O_affine", "E5O_rich", "E7_deepwd"]):
        e2.show(f"E8 {case}", run(case))
