"""E10: choosing lambda. The paper's CV (held-out gradient norm) picks lambda ~ 1e-3 almost everywhere,
while larger lambda often covers better. Compared rules:
  lam0    lambda = 0 (plain quantile regression over the features)
  cv      the paper's rule (our implementation, lrqr_cv_lambda)
  wcov    3-fold on calibration: estimate target coverage of each lambda on the held-out fold with
          weights from a logistic density-ratio estimate; pick the narrowest lambda (mean width on
          the unlabeled target set) whose estimated coverage is >= 1 - alpha, else the highest.
  oracle  narrowest lambda whose TEST coverage is >= 1 - alpha (not available in practice).
Run one setting per process:  python exp10_lambda_rule.py E2_d1 | E2_d2 | E2_d3 | E5O | air_T2
"""
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lrqr import fit_lrqr, estimate_ratio_logistic, lrqr_cv_lambda
import exp2_stress as e2
import exp5_screening as e5

alpha = 0.1
LAM_GRID = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def data(setting, rep):
    if setting.startswith("E2"):
        d, delta = 10, float(setting[-1])
        rng = np.random.default_rng(1000 * rep + d)
        xt, yt = e2.sample(e2.N_TRAIN, d, 0.0, rng); f = LinearRegression().fit(xt, yt)
        xc, yc = e2.sample(e2.N1, d, 0.0, rng); xs, _ = e2.sample(e2.N3, d, 0.0, rng)
        xg, _ = e2.sample(e2.N2, d, delta, rng); xe, ye = e2.sample(e2.M, d, delta, rng)
        phi = lambda x: np.c_[np.ones(len(x)), x, x ** 2, np.abs(x)]
        zc, zs, zg = xc, xs, xg
    elif setting == "E5O":
        d = e5.d
        rng = np.random.default_rng(1000 * rep + 7)
        xt = rng.normal(size=(e5.N_TRAIN, d)); f = LinearRegression().fit(xt, e5.response(xt, rng))
        xc = rng.normal(size=(e5.N1, d)); yc = e5.response(xc, rng)
        xs = rng.normal(size=(e5.N3, d))
        draw = lambda n: rng.normal(size=(n, d)) + rng.uniform(0, 3, size=(n, 1)) * e5.w
        xg = draw(e5.N2); xe = draw(e5.M); ye = e5.response(xe, rng)
        phi = lambda x: np.c_[np.ones(len(x)), x, np.abs(f.predict(x))]
        zc, zs, zg = xc, xs, xg
    else:                                   # airfoil, tilt T2 (towards high angle of attack)
        import exp9_airfoil as a
        rng = np.random.default_rng(rep)
        idx = rng.permutation(len(a.Y)); n = len(a.Y)
        tr, ca, po = idx[: n // 2], idx[n // 2: 3 * n // 4], idx[3 * n // 4:]
        f = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=rep).fit(a.X[tr], a.Y[tr])
        lp = 0.3 * a.X[po, 1]; p = np.exp(lp - lp.max()); p /= p.sum()
        tg = rng.choice(po, len(po), p=p); te = rng.choice(po, len(po), p=p)
        xc, yc, xs, xg, xe, ye = a.X[ca], a.Y[ca], a.X[tr], a.X[tg], a.X[te], a.Y[te]
        mu, sd = a.X.mean(0), a.X.std(0)
        phi = lambda x: np.c_[np.ones(len(x)), (x - mu) / sd, ((x - mu) / sd) ** 2]
        zc, zs, zg = (xc - mu) / sd, (xs - mu) / sd, (xg - mu) / sd
    sc = np.abs(yc - f.predict(xc)); se = np.abs(ye - f.predict(xe))
    return phi(xc), sc, phi(xs), phi(xg), phi(xe), se, zc, zs, zg


def wcov_lambda(Pc, sc, Ps, Pg, wc, seed):
    folds = np.random.default_rng(seed).permutation(len(sc)) % 3
    est_cov, est_wid = [], []
    for lam in LAM_GRID:
        hit, wsum, wid = 0.0, 0.0, []
        for k in range(3):
            tr, ho = folds != k, folds == k
            th, _ = fit_lrqr(Pc[tr], sc[tr], Ps, Pg, alpha, lam)
            hit += np.sum(wc[ho] * (sc[ho] <= Pc[ho] @ th)); wsum += wc[ho].sum()
            wid.append(np.mean(np.maximum(Pg @ th, 0)))
        est_cov.append(hit / wsum); est_wid.append(np.mean(wid))
    est_cov, est_wid = np.array(est_cov), np.array(est_wid)
    ok = np.where(est_cov >= 1 - alpha)[0]
    return LAM_GRID[ok[np.argmin(est_wid[ok])]] if len(ok) else LAM_GRID[int(np.argmax(est_cov))]


if __name__ == "__main__":
    setting = sys.argv[1]
    reps = 30 if setting.startswith("air") else 10
    res = {k: ([], [], []) for k in ["lam0", "cv", "wcov", "oracle"]}
    for rep in range(reps):
        Pc, sc, Ps, Pg, Pe, se, zc, zs, zg = data(setting, rep)
        fits = {lam: fit_lrqr(Pc, sc, Ps, Pg, alpha, lam)[0] for lam in LAM_GRID}
        cov = {lam: np.mean(se <= Pe @ th) for lam, th in fits.items()}
        wid = {lam: np.mean(2 * np.maximum(Pe @ th, 0)) for lam, th in fits.items()}
        ok = [l for l in LAM_GRID if cov[l] >= 1 - alpha]
        chosen = {
            "lam0": 0.0,
            "cv": lrqr_cv_lambda(Pc, sc, Ps, Pg, alpha, LAM_GRID, seed=rep)[0],
            "wcov": wcov_lambda(Pc, sc, Ps, Pg, estimate_ratio_logistic(zs, zg)(zc), seed=rep),
            "oracle": min(ok, key=lambda l: wid[l]) if ok else max(LAM_GRID, key=lambda l: cov[l]),
        }
        for k, lam in chosen.items():
            res[k][0].append(cov[lam]); res[k][1].append(wid[lam]); res[k][2].append(lam)
        print(f"  rep {rep} done", flush=True)
    print(f"== E10 lambda rules, {setting}, {reps} reps")
    for k, (c, w, l) in res.items():
        print(f"   {k:7s} cov {np.mean(c):.3f} +- {np.std(c):.3f}   width {np.median(w):7.3f}   "
              f"median lambda {np.median(l):.3g}")
