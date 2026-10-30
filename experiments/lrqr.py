"""Reference implementation of LR-QR (Joshi et al., NeurIPS 2025) and baselines.

The authors' code is not public (github.com/shayankiyani98/LR-QR only has a README,
checked 2026-10-28), so this is our own implementation from the paper.

Empirical LR-QR objective, over thresholds h(x) = Phi(x) @ theta and a scalar beta:

    J(theta, beta) = mean_cal[ pinball_{1-alpha}(S - h(X)) ]
                     + lam * ( beta^2 * mean_src[h^2] - 2 * beta * mean_tgt[h] )

For fixed theta the beta-term is a 1-D quadratic, minimised at beta* = b / a with
a = mean_src[h^2], b = mean_tgt[h]. Plugging it in gives

    J(theta) = mean_cal[pinball] - lam * max(b, 0)^2 / a,

the same reparametrisation the paper uses in its proofs, which we minimise with L-BFGS.
Prediction set: {y : S(x, y) <= h(x)}.

Assumptions where the paper is unspecific (marked [A1]-[A6] in the code):
  [A1] beta >= 0 with no upper bound. The paper's constrained problem uses beta in
       [beta_min, beta_max] with beta_min > 0, but gives no values for practice.
  [A2] No norm bound B on theta. The paper uses a B-ball, again without values.
  [A3] Optimiser: L-BFGS on a softplus-smoothed pinball loss (width delta = 1e-4).
       The paper does not name its optimiser.
  [A4] lambda grid: lambda* = c * n1^(-1/3) * (1/n2 + 1/n3)^(-1/3) (paper eq. 4) with
       c = 1; the constant is not given. Experiments E1-E12 used a fixed grid
       [0, 1e-3, ..., 1] instead (see paper_lambda_grid).
  [A5] CV validation measure: 2-norm of the gradient in theta only (beta fixed at its
       fitted value), with the exact pinball subgradient, averaged over held-out points.
       The paper says "the gradient of the LR-QR objective" without specifying.
  [A6] CV folds are drawn separately in the calibration, unlabeled source and unlabeled
       target sets. The paper folds "the combined calibration and unlabeled target data".
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------- pinball loss
def _pinball_smooth(u, tau, delta):
    """Smoothed pinball loss rho_tau(u), u = s - h; and its derivative in u."""
    loss = tau * u + delta * np.logaddexp(0.0, -u / delta)
    dloss = tau - expit(-u / delta)
    return loss, dloss


# ---------------------------------------------------------------- LR-QR
def lrqr_objective(theta, phi_cal, s_cal, phi_src, mean_tgt_phi, tau, lam, delta):
    """beta-eliminated LR-QR objective J(theta) and its gradient."""
    h_cal = phi_cal @ theta
    loss, dloss = _pinball_smooth(s_cal - h_cal, tau, delta)
    J = loss.mean()
    g = -(dloss[:, None] * phi_cal).mean(0)
    if lam > 0:
        h_src = phi_src @ theta
        a = np.mean(h_src ** 2) + 1e-12
        b = mean_tgt_phi @ theta
        if b > 0:                           # [A1] beta* = max(b, 0) / a, no upper bound
            J -= lam * b * b / a
            grad_a = 2.0 * (h_src[:, None] * phi_src).mean(0)
            g -= lam * (2.0 * b * mean_tgt_phi / a - b * b * grad_a / a ** 2)
    return J, g


def fit_lrqr(phi_cal, s_cal, phi_src, phi_tgt, alpha, lam, delta=1e-4, theta0=None):
    """Fit LR-QR. Returns (theta, beta). lam = 0 gives plain linear quantile regression.
    phi_cal / s_cal: labeled calibration (source) data; phi_src: unlabeled source;
    phi_tgt: unlabeled target. phi[:, 0] must be the constant feature (used to initialise)."""
    tau = 1.0 - alpha
    mean_tgt_phi = phi_tgt.mean(0)
    if theta0 is None:
        theta0 = np.zeros(phi_cal.shape[1])
        theta0[0] = np.quantile(s_cal, tau)
    # [A2] unconstrained theta (no B-ball); [A3] L-BFGS on the smoothed objective
    res = minimize(lrqr_objective, theta0, args=(phi_cal, s_cal, phi_src, mean_tgt_phi, tau, lam, delta),
                   jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-10})
    theta = res.x
    h_src = phi_src @ theta
    beta = max(mean_tgt_phi @ theta, 0.0) / (np.mean(h_src ** 2) + 1e-12)   # [A1]
    return theta, beta


def paper_lambda_grid(n1, n2, n3, c=1.0, size=10):
    """The paper's grid (Sec. 5.1): `size` uniform points from lambda*/10 to lambda*, with
    lambda* = c * n1^(-1/3) * (1/n2 + 1/n3)^(-1/3) from eq. (4). [A4] c is not given; c = 1."""
    lam_star = c * n1 ** (-1 / 3) * (1 / n2 + 1 / n3) ** (-1 / 3)
    return list(np.linspace(lam_star / 10, lam_star, size))


def lrqr_cv_lambda(phi_cal, s_cal, phi_src, phi_tgt, alpha, lam_grid, n_folds=3, seed=0):
    """The paper's lambda selection (Sec. 5.1, as reported): K-fold CV that picks the
    lambda with the smallest held-out norm of the LR-QR objective's gradient."""
    rng = np.random.default_rng(seed)
    # [A6] separate folds in calibration, unlabeled source and unlabeled target data
    folds_c = np.array_split(rng.permutation(len(s_cal)), n_folds)
    folds_t = np.array_split(rng.permutation(len(phi_tgt)), n_folds)
    folds_s = np.array_split(rng.permutation(len(phi_src)), n_folds)
    tau = 1.0 - alpha
    scores = []
    for lam in lam_grid:
        vals = []
        for k in range(n_folds):
            tr = lambda f: np.concatenate([f[j] for j in range(n_folds) if j != k])
            th, be = fit_lrqr(phi_cal[tr(folds_c)], s_cal[tr(folds_c)], phi_src[tr(folds_s)],
                              phi_tgt[tr(folds_t)], alpha, lam)
            pc, sc = phi_cal[folds_c[k]], s_cal[folds_c[k]]
            ps, pt = phi_src[folds_s[k]], phi_tgt[folds_t[k]]
            # [A5] gradient in theta only, beta fixed, exact subgradient, mean over held-out points
            dl = (sc <= pc @ th).astype(float) - tau
            g = (dl[:, None] * pc).mean(0) + lam * (2 * be ** 2 * ((ps @ th)[:, None] * ps).mean(0)
                                                    - 2 * be * pt.mean(0))
            vals.append(np.linalg.norm(g))
        scores.append(np.mean(vals))
    return lam_grid[int(np.argmin(scores))], np.array(scores)


# ---------------------------------------------------------------- baselines
def split_cp_threshold(s_cal, alpha):
    n = len(s_cal)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(s_cal)[k - 1]


def weighted_cp_thresholds(s_cal, w_cal, w_test, alpha):
    """Weighted split CP (Tibshirani et al. 2019): per-test-point threshold.
    w_cal: (n,) likelihood ratios at calibration points; w_test: (m,) at test points."""
    order = np.argsort(s_cal)
    s_sorted, w_sorted = s_cal[order], w_cal[order]
    cw = np.cumsum(w_sorted)
    total = cw[-1]
    out = np.empty(len(w_test))
    for j, wt in enumerate(w_test):
        p = cw / (total + wt)                      # cumulative mass of calibration points
        idx = np.searchsorted(p, 1 - alpha)        # first index reaching 1 - alpha
        out[j] = np.inf if idx >= len(s_sorted) else s_sorted[idx]
    return out


def estimate_ratio_logistic(x_src, x_tgt, C=1.0):
    """Classifier-based density-ratio estimate r(x) = p(tgt|x)/p(src|x) * n_src/n_tgt."""
    X = np.vstack([x_src, x_tgt])
    z = np.r_[np.zeros(len(x_src)), np.ones(len(x_tgt))]
    clf = LogisticRegression(C=C, max_iter=5000).fit(X, z)
    prior = len(x_src) / len(x_tgt)

    def r(x):
        logit = clf.decision_function(x)
        return np.exp(np.clip(logit, -700, 700)) * prior
    return r


def ess(w):
    """Effective sample size (sum w)^2 / sum w^2."""
    w = np.asarray(w, float)
    return w.sum() ** 2 / np.sum(w ** 2)
