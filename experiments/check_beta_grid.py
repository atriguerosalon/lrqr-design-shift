"""Check whether the large-lambda coverage failure (E2, rich features, strong shift) is an
optimisation artefact of eliminating beta (non-convex) or a property of the objective.

For fixed beta, the LR-QR objective is convex in theta (pinball + convex quadratic + linear), so we
solve it per beta on a grid and keep the best overall objective. We also report the result when
beta is restricted to a bounded interval, as in the paper's constrained problem.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from lrqr import _pinball_smooth, fit_lrqr

alpha, delta, d = 0.1, 3.0, 10
tau = 1 - alpha

def sample(n, shift, rng):
    x = rng.normal(size=(n, d)); x[:, 0] += shift
    y = x[:, 1] + (0.1 + 0.5 * np.abs(x[:, 0])) * rng.normal(size=n)
    return x, y

phi = lambda x: np.c_[np.ones(len(x)), x, x ** 2, np.abs(x)]

def fit_fixed_beta(Pc, sc, Ps, Pg, lam, beta, th0):
    mg = Pg.mean(0)
    def obj(th):
        u = sc - Pc @ th
        loss, dl = _pinball_smooth(u, tau, 1e-4)
        hs = Ps @ th
        J = loss.mean() + lam * (beta ** 2 * np.mean(hs ** 2) - 2 * beta * mg @ th)
        g = -(dl[:, None] * Pc).mean(0) + lam * (2 * beta ** 2 * (hs[:, None] * Ps).mean(0) - 2 * beta * mg)
        return J, g
    r = minimize(obj, th0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-10})
    return r.x, r.fun

for lam in [0.01, 0.1, 1.0]:
    out = {"eliminated": [], "grid_free": [], "grid_beta<=1": []}
    for rep in range(5):
        rng = np.random.default_rng(1000 * rep + d)
        xt, yt = sample(5000, 0.0, rng); f = LinearRegression().fit(xt, yt)
        xc, yc = sample(2000, 0.0, rng); sc = np.abs(yc - f.predict(xc))
        xs, _ = sample(2000, 0.0, rng); xg, _ = sample(5000, delta, rng)
        xe, ye = sample(5000, delta, rng); se = np.abs(ye - f.predict(xe))
        Pc, Ps, Pg, Pe = phi(xc), phi(xs), phi(xg), phi(xe)
        th0, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, 0.0)
        th_e, _ = fit_lrqr(Pc, sc, Ps, Pg, alpha, lam, theta0=th0)
        out["eliminated"].append(np.mean(se <= Pe @ th_e))
        best, best_b1 = (np.inf, None), (np.inf, None)
        for beta in np.logspace(-3, 2, 26):
            th, J = fit_fixed_beta(Pc, sc, Ps, Pg, lam, beta, th0)
            if J < best[0]: best = (J, th)
            if beta <= 1.0 and J < best_b1[0]: best_b1 = (J, th)
        out["grid_free"].append(np.mean(se <= Pe @ best[1]))
        out["grid_beta<=1"].append(np.mean(se <= Pe @ best_b1[1]))
    print(f"lambda={lam}: " + "  ".join(f"{k} {np.mean(v):.3f}" for k, v in out.items()))
