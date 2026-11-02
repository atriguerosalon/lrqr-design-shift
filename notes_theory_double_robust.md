# LR-QR is doubly robust at the population level (exact coverage identity)

Working note, 29 Oct 2026. Numerical check: experiments/exp12_decomposition.py, out_E12_*.txt.

## Setup

- Covariate shift: P(S | X) is the same under P1 (source) and P2 (target). r = dP2/dP1, E1[r] = 1.
- τ = 1 − α. Thresholds h ∈ H = span(φ1, …, φd). **Assume H contains the constants** (all our feature maps do).
- Conditional miscoverage of a threshold: m_h(x) = P(S ≤ h(x) | X = x) − τ. It is the same function under P1 and P2.
- Π = L2(P1) projection onto H. r_H = Πr, m⊥ = m_h − Π m_h.
- LR-QR population objective: L(h, β) = E1[ℓ_τ(S − h)] + λ(β² E1[h²] − 2β E2[h]).

## Step 1: an identity for any h

P2(S ≤ h(X)) − τ = E2[m_h] = E1[r m_h] = E1[r_H m_h] + E1[(r − r_H) m_h].

Because r − r_H ⊥ H, the second term equals E1[(r − r_H) m⊥]. Only the part of the miscoverage that H cannot represent interacts with the part of the ratio that H cannot represent.

## Step 2: stationarity of LR-QR

Assume P(S ≤ · | x) is continuous and (h, β) is a stationary point with β > 0.

- **∂β:** β E1[h²] = E2[h] = E1[r h].
- **∂h in direction g ∈ H:** E1[g m_h] + 2λβ E1[g(βh − r)] = 0, i.e. E1[g m_h] = 2λβ E1[g(r − βh)].
- **Take g = r_H:** E1[r_H m_h] = 2λβ E1[r_H(r_H − βh)] (since r − r_H ⊥ r_H).
- **Use ∂β again:** E1[h(r_H − βh)] = E1[h(r − βh)] = 0. Therefore E1[r_H m_h] = 2λβ E1[(r_H − βh)²].

This is the paper's Prop. 4.1 (Appendix J), rearranged.

## Result

**Exact population coverage.** At any stationary point of LR-QR,

  P2(S ≤ h(X)) = τ + 2λβ E1[(r_H − βh)²] + E1[(r − r_H) · m⊥(X)].

- The middle term is nonnegative, and is the paper's over-coverage term.
- The last term is the misspecification term. By Cauchy–Schwarz:

  |E1[(r − r_H) m⊥]| ≤ ‖r − r_H‖₂ · ‖m⊥‖₂.

- **Comparison with the paper.** Thm 4.3 (unchanged in arXiv v5, Jul 2025) instead pays (1 − α) E1|r − r_B|. That is the worst case over all miscoverage functions m ∈ [−τ, 1 − τ], independent of how good h is. Our bound gives it back as a special case, since |m| ≤ τ for τ ≥ 1/2.
- **Double robustness.** The misspecification term vanishes if r ∈ H, **or** if m_h ∈ H. In particular it vanishes if h is the true conditional τ-quantile of S (then m ≡ 0).
  - The paper uses "doubly robust" only for Qiu et al. (2023) and Yang et al. (2024), and presents LR-QR as an alternative.
  - Here the ratio "error" is a projection error, not an estimation error.
- **The case λ = 0** (Gibbs–Cherian–Candès without imputation) is the same formula without the middle term.
- **What Gibbs et al. (JRSSB 2025, arXiv 2305.12616) do** (from a text search, not a full read):
  - They treat shifts outside the unpenalized class through an RKHS penalty. Thm 3 / eq. (3.3) and Prop. 2 give the coverage error under f as 2λ E⟨ĝ_K, f_K⟩ plus an interpolation term, estimable by plug-in when f is known.
  - No product (ratio residual × conditional miscoverage) form was found, and no statement that a correct quantile makes coverage hold for every shift.
  - The identity is short algebra, so it may be folklore. Ask Francesco before calling it new.

## Consequences for the project

1. **Feature design should target the score, not the ratio.**
   - The paper's advice is "enlarge H to capture r". That is hard in high dimension, which is the reason LR-QR exists.
   - The identity says it is enough to capture the conditional quantile structure of the score. One learned feature g(x) ≈ E[S | x] often does that; this is the E8 / E9 fix.
2. **λ cannot fix misspecification.** λ enters the misspecification term only through h. Pulling βh toward r_H does not shrink m⊥, which is consistent with E10 (λ is not the bottleneck).
3. **A source-only diagnostic.**
   - ‖m⊥‖ can be estimated from labeled source data: regress 1{S ≤ h(X)} on x with a flexible, cross-fitted model, subtract τ, and project out φ. This is D̂ in E12.
   - The E8 "score skill" check is a crude proxy for the same quantity.
4. **A sensitivity certificate.**
   - ‖r − r_H‖² = E1[r²] − ‖r_H‖². The projection r_H = φᵀ Σ1⁻¹ E2[φ] is estimable from unlabeled data at the parametric rate. E1[r²] = 1 + χ²(P2‖P1) is not.
   - Hence for any target with χ²(P2‖P1) ≤ ρ:

     coverage ≥ τ + (λ term) − sqrt(1 + ρ − ‖r_H‖²) · ‖m⊥‖ (− finite-sample terms).

   - The user picks ρ, and the data give the rest. This is a possible "fix" contribution: it turns LR-QR's silent failure into a quantified one.

## Caveats

- **Population statement.** Finite samples add the paper's E_cov-type terms. The empirical stationarity conditions give the same identity with empirical E1, E2 and an extra generalization error. E12 measures the leftover in-class term directly.
- **Constant feature required.** The statement needs H to contain constants. The paper's constrained H_B (a ball) does not give exact orthogonality to constants, which is probably why its bound is cruder.
- **D̂ is a nonparametric estimate** of a conditional-coverage function. It is noisy in high dimension, and noise biases it upward (conservative).
- **Unproven assumptions.**
  - β > 0 at the stationary point.
  - Continuity of P(S ≤ · | x).
  - The smoothed pinball loss in our code only approximates the stationarity conditions.
