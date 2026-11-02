"""E12: an exact decomposition of LR-QR's coverage gap, checked numerically.

Let tau = 1 - alpha, m(x) = P(S <= h(x) | x) - tau (conditional miscoverage of the threshold h),
r_H = L2(P1)-projection of the true ratio r onto H = span(phi). If H contains constants, then
E1[r - r_H] = 0, so for ANY threshold h

    P2(S <= h(X)) - tau = E1[r m] = E1[r_H m]  +  E1[(r - r_H) m].
                                    (in-class)    (misspecification)

At a population stationary point of LR-QR the in-class term equals 2 lam beta E1[(r_H - beta h)^2] >= 0
(the paper's Prop. 4.1, rearranged). The misspecification term is a product: since r - r_H is
orthogonal to H, E1[(r - r_H) m] = E1[(r - r_H) m_perp] with m_perp = m - Proj_H m, so

    |misspec| <= ||r - r_H||_2 * ||m_perp||_2,

whereas Theorem 4.3 of the paper pays tau * E1|r - r_H| regardless of h. The product is zero when
either r is in H or h is a correct conditional quantile up to H (double robustness).

Here r and the law of S given x are known in closed form, so every term is computed by Monte Carlo
on 200k fresh P1 points. D_hat is a calibration-only estimate of ||m_perp||: cross-fitted gradient
boosting classifier for 1{S <= h(X)}, minus tau, with the projection onto phi removed.
Run one case per process: python exp12_decomposition.py E4_blind | E4_affine | E4_rich | E5O_affine | E5O_rich
"""
import sys
import numpy as np
from scipy.stats import norm
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from lrqr import fit_lrqr, lrqr_cv_lambda
import exp5_screening as e5
from exp8_check_and_fix import make_case

alpha, tau = 0.1, 0.9
N_MC = 200_000
LAM_GRID = [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]


def truth(case, x, f):
    """True ratio r(x) and conditional mean / sd of Y given x under the case's model."""
    if case.startswith("E4"):
        delta = 1.5
        return np.exp(delta * x[:, 0] - delta ** 2 / 2), x[:, 1], 0.1 + 0.5 * np.abs(x[:, 0])
    z = x @ e5.w                            # x + s w, s ~ U(0, 3), |w| = 1
    r = np.exp(z ** 2 / 2) * np.sqrt(2 * np.pi) * (norm.cdf(3 - z) - norm.cdf(-z)) / 3
    return r, z, 0.1 + 0.5 * np.abs(z)


def proj(P, v):
    return P @ np.linalg.lstsq(P, v, rcond=None)[0]


def d_hat(xc, Pc, sc, h_c, seed):
    """Calibration-only estimate of ||m_perp||_2 (2-fold cross-fitting)."""
    ind = (sc <= h_c).astype(int)
    folds = np.random.default_rng(seed).permutation(len(sc)) % 2
    p = np.empty(len(sc))
    for k in range(2):
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                             random_state=seed).fit(xc[folds != k], ind[folds != k])
        p[folds == k] = clf.predict_proba(xc[folds == k])[:, 1]
    mh = p - tau
    return np.sqrt(np.mean((mh - proj(Pc, mh)) ** 2))


def run(case, reps=5):
    base = {"E4_rich": "E4_affine"}.get(case, case)
    rows = []
    for rep in range(reps):
        D = make_case(base, rep)
        if case == "E4_rich":
            D["phi"] = lambda x: np.c_[np.ones(len(x)), x, x ** 2, np.abs(x)]
        rng_mc = np.random.default_rng(10 ** 6 + rep)
        d = D["xc"].shape[1]
        xm = rng_mc.normal(size=(N_MC, d))
        xm2 = rng_mc.normal(size=(N_MC, d))            # fresh P2 points, for a direct check of the gap
        if case.startswith("E4"):
            xm2[:, 0] += 1.5
        else:
            xm2 += rng_mc.uniform(0, 3, size=(N_MC, 1)) * e5.w
        maps = {"base": D["phi"]}
        if case != "E4_rich" and not case.endswith("rich"):
            g = MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=1000, early_stopping=True,
                             random_state=rep).fit(D["xa"], D["sa"])
            maps["+g"] = lambda x, g=g: np.c_[D["phi"](x), g.predict(x)]
        r, mu, sd = truth(case, xm, None)
        fm = D["f"].predict(xm)
        _, mu2, sd2 = truth(case, xm2, None); fm2 = D["f"].predict(xm2)
        for name, phi in maps.items():
            Pc, Ps, Pg, Pm, Pm2 = phi(D["xc"]), phi(D["xs"]), phi(D["xg"]), phi(xm), phi(xm2)
            Pe = phi(D["xe"])
            th0, _ = fit_lrqr(Pc, D["sc"], Ps, Pg, alpha, 0.0)
            lam_cv, _ = lrqr_cv_lambda(Pc, D["sc"], Ps, Pg, alpha, LAM_GRID, seed=rep)
            for lab, lam in [("lam0", 0.0), ("cv", lam_cv)]:
                th, beta = (th0, 0.0) if lam == 0 else fit_lrqr(Pc, D["sc"], Ps, Pg, alpha, lam, theta0=th0)
                h = np.maximum(Pm @ th, 0)
                m = norm.cdf((fm + h - mu) / sd) - norm.cdf((fm - h - mu) / sd) - tau
                h2 = np.maximum(Pm2 @ th, 0)
                gap_p2 = np.mean(norm.cdf((fm2 + h2 - mu2) / sd2) - norm.cdf((fm2 - h2 - mu2) / sd2)) - tau
                rH = proj(Pm, r); m_perp = m - proj(Pm, m)
                rows.append(dict(
                    feat=name, lam=lab,
                    test_gap=np.mean(D["se"] <= Pe @ th) - tau, gap_p2=gap_p2,
                    gap=np.mean(r * m), inclass=np.mean(rH * m), misspec=np.mean((r - rH) * m),
                    lam_term=2 * lam * beta * np.mean((rH - beta * Pm @ th) ** 2),
                    paper_pen=tau * np.mean(np.abs(r - rH)),
                    r_res=np.sqrt(np.mean((r - rH) ** 2)), m_perp=np.sqrt(np.mean(m_perp ** 2)),
                    prod_pen=np.sqrt(np.mean((r - rH) ** 2)) * np.sqrt(np.mean(m_perp ** 2)),
                    d_hat=d_hat(D["xc"], Pc, D["sc"], Pc @ th, rep)))
        print(f"  {case} rep {rep} done", flush=True)
    return rows


if __name__ == "__main__":
    case = sys.argv[1]
    rows = run(case)
    keys = ["test_gap", "gap_p2", "gap", "inclass", "lam_term", "misspec", "paper_pen", "prod_pen", "r_res", "m_perp", "d_hat"]
    print(f"== E12 {case}, 5 reps (means). gap = E1[r m] = inclass + misspec; gap_p2 = E2[m] and test_gap = checks")
    print("   feat  lam   " + " ".join(f"{k:>9s}" for k in keys))
    for feat in dict.fromkeys(rw["feat"] for rw in rows):
        for lab in ["lam0", "cv"]:
            sel = [rw for rw in rows if rw["feat"] == feat and rw["lam"] == lab]
            print(f"   {feat:5s} {lab:5s} " + " ".join(f"{np.mean([rw[k] for rw in sel]):9.4f}" for k in keys))
