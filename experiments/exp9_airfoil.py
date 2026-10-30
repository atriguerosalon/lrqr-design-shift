"""E9: real data. NASA airfoil self-noise (UCI, 1503 wind-tunnel points).

Inputs: frequency [Hz], angle of attack [deg], chord [m], free-stream velocity [m/s], suction-side
displacement thickness [m]. Output: scaled sound pressure level [dB].
Frequency and thickness are log-transformed (as in Tibshirani et al. 2019).

Per repetition: random split into train 50% (surrogate: random forest), calibration 25%, and a
target pool 25%. Target inputs are drawn from the pool with probability proportional to w(x):
  T1  exp(x.beta), beta = (-1, 0, 0, 0, 1) on the transformed inputs (Tibshirani et al. style tilt)
  T2  exp(0.3 * angle of attack)      towards high angle of attack (near stall)
  T3  exp(-0.3 * f(x))                towards candidates the surrogate predicts to be quiet
Two independent draws: one unlabeled target set for LR-QR / the ratio classifier, one test set.
The true ratio is w(x) up to a constant, so weighted CP with the true r is available as a reference.
Fix from E8: g(x) = gradient boosting fit to out-of-bag forest residuals on the training split.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from lrqr import fit_lrqr, split_cp_threshold, estimate_ratio_logistic, ess, lrqr_cv_lambda
from exp2_stress import wcp, show

alpha = 0.1
LAMS = [1e-2, 1e-1, 1.0]
LAM_GRID = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
raw = np.loadtxt("../data/uci_airfoil/airfoil_self_noise.dat")
X = raw[:, :5].copy(); X[:, 0] = np.log(X[:, 0]); X[:, 4] = np.log(X[:, 4])
Y = raw[:, 5]
Z = (X - X.mean(0)) / X.std(0)                      # standardized inputs for features / classifier


def run(tilt, reps=50):
    rows = {}
    def add(k, cov, wid):
        rows.setdefault(k, [[], []]); rows[k][0].append(cov); rows[k][1].append(wid)
    for rep in range(reps):
        rng = np.random.default_rng(rep)
        idx = rng.permutation(len(Y)); n = len(Y)
        tr, ca, po = idx[: n // 2], idx[n // 2: 3 * n // 4], idx[3 * n // 4:]
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, oob_score=True,
                                   random_state=rep, n_jobs=1).fit(X[tr], Y[tr])
        pred = rf.predict
        logw = {"T1": lambda i: X[i] @ np.array([-1.0, 0, 0, 0, 1.0]),
                "T2": lambda i: 0.3 * X[i, 1],
                "T3": lambda i: -0.3 * pred(X[i])}[tilt]
        lp = logw(po); p = np.exp(lp - lp.max()); p /= p.sum()
        tg = rng.choice(po, len(po), p=p); te = rng.choice(po, len(po), p=p)
        sc = np.abs(Y[ca] - pred(X[ca])); se = np.abs(Y[te] - pred(X[te]))

        q = split_cp_threshold(sc, alpha); add("split", np.mean(se <= q), 2 * q)
        th = wcp(sc, logw(ca), logw(te), alpha)
        add("wcp_true_r", np.mean(se <= th), np.median(2 * th))
        lc = logw(ca); rows.setdefault("_ess_true", []).append(ess(np.exp(lc - lc.max())))
        rhat = estimate_ratio_logistic(Z[tr], Z[tg])
        th = wcp(sc, np.log(np.maximum(rhat(Z[ca]), 1e-300)), np.log(np.maximum(rhat(Z[te]), 1e-300)), alpha)
        add("wcp_logit_r", np.mean(se <= th), np.median(2 * th))

        g = HistGradientBoostingRegressor(random_state=rep).fit(X[tr], np.abs(Y[tr] - rf.oob_prediction_))
        maps = {"base": lambda i: np.c_[np.ones(len(i)), Z[i], Z[i] ** 2],
                "+g": lambda i: np.c_[np.ones(len(i)), Z[i], Z[i] ** 2, g.predict(X[i])]}
        for name, phi in maps.items():
            Pc, Ps, Pg, Pe = phi(ca), phi(tr), phi(tg), phi(te)
            th0, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, 0.0)
            h = Pe @ th0; add(f"{name}_lam0", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
            if name == "base":
                for l in LAMS:
                    thl, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, l, theta0=th0)
                    h = Pe @ thl; add(f"{name}_lam{l:g}", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
            lam_cv, _ = lrqr_cv_lambda(Pc, sc, Ps, Pg, alpha, LAM_GRID, seed=rep)
            thl, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, lam_cv, theta0=th0)
            h = Pe @ thl; add(f"{name}_cv", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
    return rows


if __name__ == "__main__":
    for tilt in ["T1", "T2", "T3"]:
        show(f"E9 airfoil self-noise, tilt {tilt}, 50 reps", run(tilt))
