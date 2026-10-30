"""E6: does LR-QR's win in E5 scenario O survive a better density-ratio estimate for weighted CP?

Same data as exp5_screening.py scenario O (same seeds and draw order). Weighted CP is run with
several ratio classifiers; the first row (logistic, C = 1) must reproduce E5's 0.811.
LR-QR with CV lambda gave 0.892 on this data (out_E5.txt).
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from exp5_screening import response, w, d, N_TRAIN, N1, N3, N2, M, alpha, S_GRID
from exp2_stress import wcp, show
from lrqr import ess


def true_logr(x):
    z = (x @ w)[:, None]
    a = S_GRID[None, :] * z - S_GRID[None, :] ** 2 / 2
    m = a.max(1, keepdims=True)
    return (m + np.log(np.exp(a - m).mean(1, keepdims=True)))[:, 0]


def ratio_from_classifier(clf, feat, xs, xg):
    X = np.r_[feat(xs), feat(xg)]
    lab = np.r_[np.zeros(len(xs)), np.ones(len(xg))]
    clf.fit(X, lab)
    def logr(x):
        p = np.clip(clf.predict_proba(feat(x))[:, 1], 1e-12, 1 - 1e-12)
        return np.log(p) - np.log1p(-p) + np.log(len(xs) / len(xg))
    return logr


ident = lambda x: x
quad = lambda x: np.c_[x, x ** 2]
oracle = lambda x: np.c_[x @ w, (x @ w) ** 2]       # knows the shift direction: reference only
ESTIMATORS = {
    "logit_C1": lambda: (LogisticRegression(C=1.0, max_iter=2000), ident),
    "logit_cv": lambda: (LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", max_iter=2000), ident),
    "logit_quad_cv": lambda: (LogisticRegressionCV(Cs=10, cv=5, scoring="neg_log_loss", max_iter=2000), quad),
    "boosting": lambda: (HistGradientBoostingClassifier(random_state=0), ident),
    "mlp": lambda: (MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, early_stopping=True,
                                  max_iter=500, random_state=0), ident),
    "oracle_dir_logit": lambda: (LogisticRegression(C=1e3, max_iter=2000), oracle),
}

if __name__ == "__main__":
    rows = {}
    def add(k, cov, wid):
        rows.setdefault(k, [[], []]); rows[k][0].append(cov); rows[k][1].append(wid)
    for rep in range(10):
        rng = np.random.default_rng(1000 * rep + 7)          # identical draw order to exp5 run("O")
        xt = rng.normal(size=(N_TRAIN, d)); f = LinearRegression().fit(xt, response(xt, rng))
        xc = rng.normal(size=(N1, d)); sc = np.abs(response(xc, rng) - f.predict(xc))
        xs = rng.normal(size=(N3, d))
        draw = lambda n: rng.normal(size=(n, d)) + rng.uniform(0, 3, size=(n, 1)) * w
        xg = draw(N2); xe = draw(M); se = np.abs(response(xe, rng) - f.predict(xe))
        sub = rng.choice(M, 1500, replace=False)
        lt = true_logr(xc)
        for name, make in ESTIMATORS.items():
            clf, feat = make()
            lr = ratio_from_classifier(clf, feat, xs, xg)
            lc = lr(xc)
            th = wcp(sc, lc, lr(xe[sub]), alpha)
            add(name, np.mean(se[sub] <= th), np.median(2 * th))
            rows.setdefault(f"_corr_{name}", []).append(np.corrcoef(lc, lt)[0, 1])
            rows.setdefault(f"_ess_{name}", []).append(ess(np.exp(lc - lc.max())))
        print(f"  rep {rep} done", flush=True)
    show("E6 ratio estimators for weighted CP, scenario O, d=50 (LR-QR CV: 0.892)", rows)
