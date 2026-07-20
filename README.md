# Readout limits of protected quantum memories — code & data

Reproduces the numerical results and figure of

> D. Henderson, *How gently can a protected quantum memory be read? Rate limits
> on logical readout from the structure of the code* (2026).

This is a self-contained reproducibility release. It contains only the code
needed to regenerate the paper's tables, figure, and analytic-lemma checks —
not the broader code-search machinery used during the research.

## Install
```
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt                      # numpy, scipy, qutip==5.3.0, matplotlib
```

## Layout
```
qudlab/codes.py              cat / GKP / number codeword & projector construction
qudlab/loss_correction.py    logical-rate ("protection") scoring
gkp_qudit2.py                finite-energy GKP codewords and jump operators
perception_frontier.py       the signal (S) and disturbance functionals; code+meter builders
perception_step3.py          spectral rate extraction
perception_ruler.py          functional validation (standalone)
autocorr_extract_d3fix.py    windowed-autocorrelation self-rate extraction (Tables I/II, incl. d>2)
round10_stageB.py            the 48 generalized-number-code search (Fig. 1b cloud)
nogo_sweep.py                meter-menu self-reporting search + exponential-floor fits (Sec. VI)
proof_gkp.py                 analytic GKP tuned-zero lemma: closed-form checks to machine precision
lemma_A_validation.py        dense 4-level model: Assumption-A remainder + flux-identity checks (App. C/D)
fig1.py                      renders Figure 1 from the tabulated values
data/                        saved outputs (JSON) — regenerate with the commands below
```

## Reproduce
```
# Analytic-lemma checks (fast, no eigensolve):
python proof_gkp.py                 # GKP tuned-zero: C(Δ), Δm, κ, Λ01 closed forms vs numerics
python lemma_A_validation.py        # Assumption-A remainder O(1/g), flux identity, g-sweep
python nogo_sweep.py                # menu no-go sweep + floor fits -> data/nogo_results.json

# Atlas self-rates (dense Liouvillian eigensolves; minutes per cell):
python autocorr_extract_d3fix.py cat3        # cat d=3  (Table I)
python autocorr_extract_d3fix.py cat4        # cat d=4  (Table I)
python autocorr_extract_d3fix.py gkp3 90     # GKP d=3 at Fock cutoff N=90 (Table I)

# The figure:
python fig1.py                      # -> readability_fig1.pdf
```

## Notes
- `qutip==5.3.0` is the version used for the paper; other 5.x releases should work
  but were not validated.
- The `paper (spectral triple-run)` reference values printed by
  `autocorr_extract_d3fix.py` are the *superseded* spectral estimates; the paper
  quotes the windowed-autocorrelation values this script computes (they differ by
  20–50% at d>2 because the spectral extractor misidentifies near-degenerate
  logical modes — see the paper's atlas section).
- The number-code protection scores use a biased recovery pump and are therefore
  conservative, as stated in the paper.
- `lemma_A_validation.py` is an independent, self-contained 4-level model. It
  confirms the *structure* of Appendix C/D: the within-code windowed rate equals
  2*gamma*kappa^2 exactly (not the spectral eigenvalue), the leak-term flux
  identity holds (to ~9% in this simplified model), which-path back-action gives
  no leading self-damage (residual O(gamma/g)), the slope stays bounded as g
  grows (no Zeno), and the block bounds hold. It is a structural check, not a
  high-precision reproduction of the paper's tightest quoted validation figures.

## License
MIT (see LICENSE).
