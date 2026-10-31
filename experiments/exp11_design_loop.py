"""E11: a design loop. Can verified designs from earlier rounds be reused for calibration?

d = 10, Y = w.x + (0.1 + 0.5 |w.x|) eps, w = 1/sqrt(d). Surrogate: random forest.
Rounds k = 1, 2, 3 push candidates further along w: x + s w, s ~ U(k-1, k). After rounds 1 and 2,
300 candidates each are verified (labelled, think CFD). We calibrate for round-3 candidates.
  A  original:   calibration = 1000 source points; surrogate trained on the training set only
  B  pooled:     calibration = source points + verified rounds 1-2; same surrogate (not retrained)
                 LR-QR's unlabeled source set is drawn from the same mixture as the pooled set
  C  pooled + retrained: surrogate retrained on training set + verified points, then the same
                 verified points reused for calibration (common practice; they are now in-sample)
  D  original + retrained: retrained surrogate, calibration on the original source points only
B and D are still covariate shift (pooled calibration is i.i.d. from a known mixture; D's
calibration points were never used for training). C breaks the independence LR-QR assumes.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from lrqr import fit_lrqr, split_cp_threshold, lrqr_cv_lambda
from exp2_stress import show

alpha, d = 0.1, 10
w = np.ones(d) / np.sqrt(d)
N_TRAIN, N_CAL, N_VER, N_SRC, N_TGT, M = 3000, 1000, 300, 2000, 3000, 3000
LAM_GRID = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def resp(x, rng):
    z = x @ w
    return z + (0.1 + 0.5 * np.abs(z)) * rng.normal(size=len(x))


def cand(n, k, rng):
    return rng.normal(size=(n, d)) + rng.uniform(k - 1, k, size=(n, 1)) * w


def rf(x, y, seed):
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=seed, n_jobs=1).fit(x, y)


if __name__ == "__main__":
    rows = {}
    def add(k, cov, wid):
        rows.setdefault(k, [[], []]); rows[k][0].append(cov); rows[k][1].append(wid)
    for rep in range(10):
        rng = np.random.default_rng(rep)
        xt = rng.normal(size=(N_TRAIN, d)); yt = resp(xt, rng)
        xc = rng.normal(size=(N_CAL, d)); yc = resp(xc, rng)
        xv = np.r_[cand(N_VER, 1, rng), cand(N_VER, 2, rng)]; yv = resp(xv, rng)
        xe = cand(M, 3, rng); ye = resp(xe, rng); xg = cand(N_TGT, 3, rng)
        src_orig = rng.normal(size=(N_SRC, d))
        n0 = int(N_SRC * N_CAL / (N_CAL + 2 * N_VER)); n1 = (N_SRC - n0) // 2
        src_pool = np.r_[rng.normal(size=(n0, d)), cand(n1, 1, rng), cand(n1, 2, rng)]
        f0 = rf(xt, yt, rep)
        f1 = rf(np.r_[xt, xv], np.r_[yt, yv], rep)
        setups = {"A_original": (f0, xc, yc, src_orig), "B_pooled": (f0, np.r_[xc, xv], np.r_[yc, yv], src_pool),
                  "C_pooled_retrained": (f1, np.r_[xc, xv], np.r_[yc, yv], src_pool),
                  "D_original_retrained": (f1, xc, yc, src_orig)}
        for name, (f, xcal, ycal, xs) in setups.items():
            sc = np.abs(ycal - f.predict(xcal)); se = np.abs(ye - f.predict(xe))
            q = split_cp_threshold(sc, alpha); add(f"{name}_split", np.mean(se <= q), 2 * q)
            phi = lambda x: np.c_[np.ones(len(x)), x, np.abs(f.predict(x))]
            Pc, Ps, Pg, Pe = phi(xcal), phi(xs), phi(xg), phi(xe)
            th0, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, 0.0)
            h = Pe @ th0; add(f"{name}_lam0", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
            lam, _ = lrqr_cv_lambda(Pc, sc, Ps, Pg, alpha, LAM_GRID, seed=rep)
            th, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, lam, theta0=th0)
            h = Pe @ th; add(f"{name}_cv", np.mean(se <= h), np.mean(2 * np.maximum(h, 0)))
            if name.startswith("C"):
                rows.setdefault("_C_verified_score_ratio", []).append(
                    np.mean(sc[N_CAL:]) / np.mean(np.abs(yv - f0.predict(xv))))
        print(f"  rep {rep} done", flush=True)
    show("E11 design loop, calibration for round-3 candidates", rows)
