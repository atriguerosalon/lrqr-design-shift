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
| `exp2_stress.py E2 / E2rich / E3 / E4` | E2-E4: shift strength, dimension, feature choice |
| `check_beta_grid.py` | Is the large-lambda failure an optimisation artefact? Effect of the beta range |
| `exp5_screening.py` | E5: top-k screening (known generator) and an optimizer push, d = 50 |
| `exp6_ratio_tuning.py` | E6: better density-ratio estimators for weighted CP |
| `exp7_deep_features.py` | E7: features from networks trained for the mean |
| `exp8_check_and_fix.py` | E8: source-only warning check and a learned score feature |
| `out_*.txt` | Raw outputs |

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

**E2, shift strength** (d = 10, error grows with |x1|, shift along x1):
- Affine features: LR-QR improves on split CP only partially at the best lambda; CV-chosen lambda
  ~1e-3 stays near split CP (e.g. delta = 1.5: split 0.736, LR-QR CV 0.746, best lambda 0.846,
  weighted CP with logistic ratio 0.888).
- Rich features (x, x^2, |x|): plain quantile regression with lambda = 0 already reaches ~0.89-0.90
  at all shift strengths; CV picks lambda ~ 0. Larger lambda breaks coverage under strong shift
  (delta = 3: lambda = 0.1 gives 0.415). Possibly prevented by the paper's beta/B bounds; to check.

**E4, features** (d = 10, delta = 1.5): features blind to x1 -> LR-QR equals split CP (0.73) for every
lambda, silently. Mean-prediction MLP features: lambda = 0.1 lifts coverage 0.80 -> 0.89.
Error-prediction MLP features: lambda = 0 already 0.89.

**E3, dimension** (delta = 1.5, rich features = 1 + 3d columns, 5 reps; n1 = 2000):
split 0.73 throughout; weighted CP (logistic) 0.90-0.91 at every d; QR lambda = 0:
0.887 / 0.891 / 0.889 / 0.846 / 0.793 for d = 5 / 20 / 50 / 100 / 200; lambda = 0.01:
0.945 / 0.941 / 0.939 / 0.924 / 0.897; lambda = 0.1: 0.947 / 0.935 / 0.885 / 0.838 / 0.665;
lambda = 1: 0.754 ... 0.488. As features approach n1, QR over-fits the calibration scores and
under-covers (paper's d << n assumption); the working lambda window narrows. Caveat: the shift is
along a single coordinate, so the logistic ratio estimate has an easy job; not yet a fair
high-dimensional test of weighted CP. ESS of true weights ~270 / 2000 at every d.

**beta range check** (`check_beta_grid.py`, delta = 3, rich features, 5 reps): the large-lambda
failure is not an optimisation artefact (beta grid with convex inner problem gives the same:
lambda = 0.1 -> 0.435 vs 0.413). But restricting beta <= 1 flips it to over-coverage (0.98).
Coverage under strong shift depends heavily on the beta interval, which the paper does not specify
in practice. Consistent with Theorem 4.3: with E1[r^2] ~ 8100 and r far from the feature span, the
misspecification term makes the bound vacuous.

**From the paper's appendix** (arXiv 2502.13030, Appendix M): Lemma M.4 sets
beta_upper = E1[r^2]^{1/2} / (B_lower * lambda_min(Sigma)^{1/2}), so the theory requires the beta
interval to grow with the shift. The unbounded-beta behaviour (undercoverage under strong shift) is
therefore the theory-compliant one; clipping beta <= 1 is an off-theory heuristic. The proof also
reparametrises LR-QR as min E1[pinball] - mu * E1[r h]^2 / E1[h^2], which is exactly our
beta-eliminated objective (E1[r h] = E2[h]), so our implementation matches the paper's own math.

**Bounded scores** (`exp2_stress.py E2rich_bounded`, S' = S/(1+S), as the theory assumes S in [0,1]):
conclusions hold. delta = 1/2/3: split 0.820/0.644/0.478; weighted CP (logistic) 0.895/0.905/0.937;
LR-QR with CV lambda 0.907/0.929/0.740. Under strong shift coverage swings 0.60-0.99 across a
factor of 10 in lambda (delta = 3: lambda 1e-3 -> 0.992 but trivial sets, 1e-2 -> 0.710,
1e-1 -> 0.614), and CV does not reliably find a good lambda.

**E5, application-shaped shifts** (`exp5_screening.py`, d = 50, shift spread over all coordinates
along w = 1/sqrt(d), errors grow with |w.x|, features [1, x, |f(x)|], 10 reps, `out_E5.txt`):
- S, top-10% screening from the calibration generator (ratio known: 10 * 1{selected}):
  split 0.670; selective CP = weighted CP with the known ratio 0.888 +- 0.029 (width 3.12);
  weighted CP with logistic ratio 0.925 +- 0.077 (width 4.24, unstable: a sigmoid cannot fit an
  indicator); QR lambda = 0 0.883; LR-QR CV 0.896 (width 3.08). LR-QR matches selective CP but
  offers nothing over it, and selective CP needs no unlabeled target set or tuning.
- O, optimizer push x + s*w, s ~ U(0, 3) (ratio unknown in practice): split 0.712; weighted CP
  true r 0.919 +- 0.051 (width 3.42); weighted CP logistic 0.811 (fails: log-ratio is not linear
  in x); QR lambda = 0 0.882; LR-QR CV 0.892 (width 3.06); lambda = 0.01 0.926; lambda = 1 0.679.
  First setting where LR-QR clearly beats the estimated-ratio baseline. But most of the gain comes
  from the |f(x)| feature (lambda = 0 is already 0.882); the regulariser adds ~0.01 at CV lambda.
  Logistic C = 1 not tuned; a better ratio classifier might close the gap (to check).

**E6, better ratio estimators for weighted CP** (`exp6_ratio_tuning.py`, same data as E5 scenario O,
10 reps, `out_E6.txt`; LR-QR CV was 0.892): logistic C = 1 0.811 (reproduces E5); logistic with CV
over C 0.808; logistic on [x, x^2] with CV 0.794; gradient boosting 0.777; MLP classifier 0.850 +- 0.055;
logistic on the true shift direction only (oracle, not available in practice) 0.907.
The learned ratios correlate well with the true one (0.78-0.94 on log scale) but are too flat:
their ESS is 460-820 against ~216 for the true ratio, so they under-weight the most shifted
calibration points and under-cover. LR-QR's win in scenario O holds against tuned estimators.

**E7, features from networks trained for the mean** (`exp7_deep_features.py`, E4 setting d = 10,
delta = 1.5, scores from each network's own predictions, 5 reps, `out_E7.txt`; split 0.732).
Ridge probe R^2 on fresh data for x1 / |x1| (|x1| drives the error size), then LR-QR coverage:
- small (32,): probe 0.92 / 0.23; lambda = 0 0.789, CV 0.826, lambda = 0.1 0.890
- deep (128,128,64): probe 0.76 / 0.49; lambda = 0 0.836, CV 0.872, lambda = 0.1 0.911
- deep, weight decay 1.0: probe 0.58 / 0.08; lambda = 0 0.768, CV 0.776, lambda = 0.1 0.856
Networks trained for the mean do not simply drop the shifted coordinate. What predicts LR-QR's
coverage is whether the features can express the error-relevant quantity |x1| (ordering of the
|x1| probe matches the ordering of CV coverage), not whether they contain x1. Strong weight decay
removes |x1| and LR-QR falls back towards split CP; CV lambda does not notice.
Possible practical check (untested): probe whether the features predict the calibration scores.

**E8, warning check and feature fix** (`exp8_check_and_fix.py`, 5 reps, `out_E8.txt`).
Check = skill of quantile regression on the features at predicting calibration scores (2-fold,
pinball loss relative to a constant quantile). Fix = append g(x), a model of the score fitted on an
extra labeled source split of 2000 points (gradient boosting or MLP).
| case | skill base | CV cov base | skill +g | CV cov +g_gbm / +g_mlp |
|---|---|---|---|---|
| E4 blind (d=10) | -0.02 | 0.733 | 0.22 | 0.842 / 0.873 |
| E4 affine (d=10) | -0.02 | 0.748 | 0.23 | 0.847 / 0.893 |
| E7 deep net, weight decay | 0.02 | 0.797 | 0.23-0.24 | 0.855 / 0.871 |
| E5 optimizer, affine (d=50) | -0.09 | 0.726 | -0.08 | 0.724 / 0.737 |
| E5 optimizer, rich (control) | 0.30 | 0.904 | 0.30 | 0.907 / 0.908 |
The check separates every failing case (skill ~ 0) from the working control (0.30). The fix
recovers most of the gap in d = 10 but stays below 0.90: g is fitted on source data and must
extrapolate into the shifted region (the MLP, which extrapolates, beats boosting, which is flat).
In d = 50 the score model cannot learn |w.x| from 2000 points, the fix does nothing, and the check
says so (skill stays ~ 0). So the check also tells you whether the fix worked.

## Summary of what the experiments suggest (not yet claims)
1. In the authors' own toy, the shift does not affect the errors (split CP is exactly valid), and
   r is far outside the affine class; it cannot show a benefit from correcting the shift.
2. With features that model the error structure, plain quantile regression (lambda = 0, Gibbs-style)
   already handles the shift; LR-QR's regulariser matters in the middle regime where features see
   the shift but do not model the error.
3. Features blind to the shift: LR-QR silently equals split CP.
4. Strong shift (large E1[r^2]): coverage is very sensitive to lambda and to the beta range; the
   paper's lambda-CV does not reliably fix it; consistent with Theorem 4.3 becoming vacuous.
5. In low dimension, weighted CP with a logistic ratio estimate is a strong baseline throughout.

## Open checks
- Ask the authors for code (email above), or Francesco whether that is appropriate.
