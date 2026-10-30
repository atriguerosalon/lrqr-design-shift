# Conformal prediction under covariate shift: where LR-QR breaks

GML course project (ETH, HS 2026). Alejandro Trigueros Alonso and Felicia Scharitzer. Supervisor: Francesco Taroni.

The anchor paper is Joshi, Kiyani, Pappas, Dobriban & Hassani, *Conformal inference under high-dimensional covariate shifts via likelihood-ratio regularization* (NeurIPS 2025, arXiv 2502.13030), which introduces LR-QR. We ask which of LR-QR's assumptions become a problem when the shift comes from surrogate-based design (optimizers, batch screening, design loops). We separate genuine weak spots from scope mismatches and try to fix one.

## Layout

- **`proposal/`**: the project proposal (`proposal.tex`, `references.bib`, logo) and the submitted PDF.
- **`notes_theory_double_robust.md`**: an exact decomposition of LR-QR's coverage gap. LR-QR covers if either the density ratio or the score's quantile structure lies in the feature span, so every failure we observed is a misspecification term.
- **`notes_screening_literature.md`**: screening from a known generator (selective conformal) versus an unknown optimizer.
- **`experiments/`**:
  - Our own LR-QR implementation (`lrqr.py`). The authors' code is not public.
  - Experiments E1–E12, with raw outputs in `out_E*.txt`.
  - `experiments/README.md` has every result and the implementation choices the paper leaves open.
- **`data/uci_airfoil/`**: NASA airfoil self-noise data from the UCI repository (CC BY 4.0), used in E9 and E10.

## Running

```
pip install -r requirements.txt
cd experiments
python exp2_stress.py        # each script prints its table; most take 5-25 min on one CPU core
```

Settings are independent, so run them as parallel processes (E10 and E12 take the setting as an argument).
