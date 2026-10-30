"""Checks of lrqr.py. Run with `python test_lrqr.py` (or pytest). Takes a few seconds.

These test that the code solves the objective it states; they cannot test the assumptions
[A1]-[A6] where the paper is unspecific (see lrqr.py).
"""
import numpy as np
from scipy.optimize import check_grad
from sklearn.linear_model import QuantileRegressor
from lrqr import (_pinball_smooth, lrqr_objective, fit_lrqr, split_cp_threshold,
                  weighted_cp_thresholds)

ALPHA, TAU = 0.1, 0.9


def shifted_data(seed, n=1500, d=10, delta=1.5):
    """Errors grow with |x1|; the target is shifted along x1 (the E2 setting)."""
    rng = np.random.default_rng(seed)
    def x(m, shift):
        z = rng.normal(size=(m, d)); z[:, 0] += shift; return z
    phi = lambda z: np.c_[np.ones(len(z)), z, z ** 2, np.abs(z)]
    xc = x(n, 0.0)
    s = np.abs((0.1 + 0.5 * np.abs(xc[:, 0])) * rng.normal(size=n))
    return phi(xc), s, phi(x(n, 0.0)), phi(x(3 * n, delta))


def pinball(u):
    return np.mean(np.maximum(TAU * u, (TAU - 1) * u))


def test_smoothed_pinball_converges_to_exact():
    u = np.linspace(-3, 3, 601)
    exact = np.maximum(TAU * u, (TAU - 1) * u)
    for delta, tol in [(1e-2, 1e-2), (1e-4, 1e-4)]:
        assert np.max(np.abs(_pinball_smooth(u, TAU, delta)[0] - exact)) <= tol


def test_gradient_matches_finite_differences():
    Pc, s, Ps, Pg = shifted_data(0, n=400)
    rng = np.random.default_rng(1)
    for lam in [0.0, 0.1, 1.0]:
        th = fit_lrqr(Pc, s, Ps, Pg, ALPHA, lam)[0] + 0.01 * rng.normal(size=Pc.shape[1])
        args = (Pc, s, Ps, Pg.mean(0), TAU, lam, 1e-2)   # wider smoothing so finite differences are meaningful
        err = check_grad(lambda t: lrqr_objective(t, *args)[0], lambda t: lrqr_objective(t, *args)[1], th)
        assert err < 1e-5, (lam, err)


def test_lambda0_matches_exact_quantile_regression():
    """With lam = 0 LR-QR is linear quantile regression, which an LP solves exactly."""
    for seed in range(3):
        Pc, s, Ps, Pg = shifted_data(seed)
        ours = fit_lrqr(Pc, s, Ps, Pg, ALPHA, 0.0)[0]
        lp = QuantileRegressor(quantile=TAU, alpha=0.0, fit_intercept=False, solver="highs").fit(Pc, s).coef_
        assert pinball(s - Pc @ ours) <= pinball(s - Pc @ lp) * (1 + 1e-3)


def test_restarts_do_not_find_a_lower_objective():
    """After eliminating beta the objective is not convex; check L-BFGS is not stuck."""
    Pc, s, Ps, Pg = shifted_data(0, delta=3.0)
    for lam in [0.01, 0.1, 1.0]:
        th = fit_lrqr(Pc, s, Ps, Pg, ALPHA, lam)[0]
        J = lambda t: lrqr_objective(t, Pc, s, Ps, Pg.mean(0), TAU, lam, 1e-4)[0]
        rng = np.random.default_rng(2)
        for _ in range(5):
            alt = fit_lrqr(Pc, s, Ps, Pg, ALPHA, lam, theta0=th + rng.normal(scale=0.3, size=len(th)))[0]
            assert J(th) <= J(alt) + 1e-4, (lam, J(th), J(alt))


def test_beta_is_the_closed_form_minimiser():
    Pc, s, Ps, Pg = shifted_data(0, n=500)
    th, beta = fit_lrqr(Pc, s, Ps, Pg, ALPHA, 0.1)
    a, b = np.mean((Ps @ th) ** 2), np.mean(Pg @ th)
    grid = np.linspace(0, 3 * beta + 1, 20001)
    assert abs(grid[np.argmin(grid ** 2 * a - 2 * grid * b)] - beta) < 1e-3


def test_weighted_cp_with_equal_weights_is_split_cp():
    s = np.random.default_rng(0).exponential(size=999)
    th = weighted_cp_thresholds(s, np.ones(999), np.ones(5), ALPHA)
    assert np.all(th == split_cp_threshold(s, ALPHA))


def test_split_cp_covers_without_shift():
    rng = np.random.default_rng(0)
    cov = [np.mean(rng.exponential(size=1000) <= split_cp_threshold(rng.exponential(size=200), ALPHA))
           for _ in range(300)]
    assert abs(np.mean(cov) - (1 - ALPHA)) < 0.01


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, f in tests:
        f()
        print("ok  ", name)
    print(f"{len(tests)} passed")
