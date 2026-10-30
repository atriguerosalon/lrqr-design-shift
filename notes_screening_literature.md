# Batch screening: literature LR-QR should be compared against (29 Oct 2026)

Why this matters: our application (screen many unlabeled candidate geometries with a surrogate,
keep the promising ones) has its own conformal literature, which neither LR-QR nor our proposal cites.

## Key observation
If candidates are drawn from the same generator as the calibration data and then filtered by the
surrogate (keep top 10%, keep predicted drag < t, ...), the selection is a known function of x.
The density ratio is then KNOWN: r(x) = 1{selected(x)} / P(selected). Weighted CP with this r is
just split CP on the calibration points that pass the same filter. No ratio estimation, no LR-QR.
This is the idea behind Bao et al. (2024) below.

LR-QR (or any learned-ratio method) is only needed when the candidate distribution itself is
unknown: e.g. candidates produced by an optimizer, a generative model, or a different design
space than the calibration set. Then the question is whether the shift is "covariate shift" at
all: if the optimizer uses the same surrogate trained on the calibration data, it is feedback
covariate shift (Fannjiang et al. 2022), and the test points depend on the training data.

## Papers (verified)
- Bao, Huo, Ren & Zou (2024). Selective conformal inference with false coverage-statement rate
  control. Biometrika 111(3), 727-742. arXiv 2301.00584. Selects on calibration and test with the
  same rule, then calibrates on the selected calibration points ("selective conditional CP").
  Exact FCR control for exchangeable selection rules. Directly the "keep the top-k" setting.
- Jin & Candes (2023). Selection by prediction with conformal p-values. JMLR 24(244), 1-41.
  Choose candidates whose outcome exceeds a threshold with FDR control. Screening, but answers
  "which ones are good", not "how wide is the interval".
- Jin & Candes (2023). Model-free selective inference under covariate shift via weighted
  conformal p-values. arXiv 2307.09291. Same as above with covariate shift (weighted CP, known
  or estimated ratio). Journal version not confirmed.
- Fannjiang, Bates, Angelopoulos, Listgarten & Jordan (2022). Conformal prediction under feedback
  covariate shift for biomolecular design. PNAS 119(43), e2204569119. The test distribution
  depends on the training data through the design algorithm. Needs the design algorithm's density
  (known for some protein design procedures, not for a black-box CFD optimizer).

## Where this leaves LR-QR for our project
1. Filter-style screening from a known generator: selective CP (Bao et al.) is simpler and exact.
   LR-QR has no reason to be used. (Testing this: exp5_screening.py, scenario S.)
2. Candidates from an unknown generator/optimizer, high-dimensional inputs: the case LR-QR is
   built for. Competitors: weighted CP with a classifier ratio (Tibshirani 2019), FCS if the design
   density is known. (Testing: exp5_screening.py, scenario O.)
3. Candidates chosen using the same surrogate that was trained on calibration-adjacent data: the
   independence LR-QR assumes may break (feedback). Out of scope for the paper, a possible extension.
Possible framing for the report: map which of the three regimes a design workflow is in, and show
LR-QR's behaviour in regime 2 against the right baselines.
