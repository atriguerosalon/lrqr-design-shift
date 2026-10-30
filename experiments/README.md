# LR-QR experiments

Own implementation of LR-QR (Joshi et al., NeurIPS 2025), because the authors' code is not
public: github.com/shayankiyani98/LR-QR only contains a README ("Codes will be published soon
... contact shayank@seas.upenn.edu"), although the rebuttal promised a release with the revision.

Requirements: see `../requirements.txt` (numpy, scipy, scikit-learn; CPU only, no torch).
Most scripts take 5-25 minutes on one core; independent settings can run as parallel processes.

## Files

| File | What |
|---|---|
| `lrqr.py` | LR-QR (beta eliminated in closed form, smoothed pinball, L-BFGS), lambda CV as described in Sec. 5.1, split CP, weighted CP, logistic density-ratio estimate, effective sample size |
| `exp1_authors_toy.py` | E1: replication of the authors' rebuttal simulation (reply to reviewer KYG7) |

## Implementation choices where the paper is unspecific

Each is tagged [A1]-[A6] in `lrqr.py`, so it is easy to find and change.
- [A1] beta: the paper's constrained problem uses beta in [beta_min, beta_max] with beta_min > 0, without
  values. We use beta >= 0 with no upper bound (closed form beta* = b/a).
- [A2] No norm bound B on theta (the paper uses a B-ball, without a value).
- [A3] Optimiser not named in the paper. We use L-BFGS on a softplus-smoothed pinball loss (width 1e-4).
- [A4] The constant in lambda* (eq. 4) is not given. `paper_lambda_grid` uses c = 1; E1-E12 used a fixed
  grid [0, 1e-3, ..., 1] instead.
- [A5] The CV measure is "the gradient of the LR-QR objective" on held-out data. We take the gradient
  in theta only, with beta fixed, averaged over held-out points.
- [A6] The paper folds "the combined calibration and unlabeled target data"; we fold the calibration,
  unlabeled source and unlabeled target sets separately.

`test_lrqr.py` (`python test_lrqr.py`, a few seconds) checks that the code solves the objective it
states: the gradient matches finite differences, lambda = 0 matches an exact LP quantile regression,
random restarts find no lower objective, beta is the closed-form minimiser, and the conformal
baselines behave as expected. It cannot check [A1]-[A6].

## Results so far (alpha = 0.1, target coverage 0.90)

**E1, authors' toy** (d = 10, homoscedastic noise, well-specified linear model, E1[r^2] = e^{4d} ~ 2e17):
split CP 0.900 (already valid: the shift does not change the error distribution); weighted CP
(true or estimated r) gives infinite intervals; LR-QR 0.893 / 0.893 / 0.909 / 0.993 / 0.997 for
lambda = 1e-6 ... 1e-2 (authors report 0.864 / 0.860 / 0.867 / 0.943 / 1.000). Same pattern, not
the exact numbers.
