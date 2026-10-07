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
perception_step3.py          spectral rate extraction; produces the self/conjugate
                             split, i.e. the Gamma_conj column of Table I
perception_ruler.py          functional validation (standalone)
autocorr_extract_d3fix.py    windowed-autocorrelation self-rate extraction (Tables I/II, incl. d>2)
round10_stageA.py            GKP rectangular-lattice aspect sweep r in {0.75, 0.85, 1.0,
                             1.18, 1.35} -> the readability/protection trade-off curve of
                             Sec. V and the lattice curve in Fig. 1b
one_r.py                     runs a single aspect ratio of round10_stageA in a fresh
                             process (needed for r > 1 on memory-limited machines;
                             see the note under Reproduce)
round10_stageB.py            the 48 generalized-number-code search (Fig. 1b cloud)
nogo_sweep.py                meter-menu self-reporting search + exponential-floor fits (Sec. VI)
proof_gkp.py                 analytic GKP tuned-zero lemma: closed-form checks to machine precision
lemma_A_validation.py        dense 4-level model: Assumption-A remainder + flux-identity checks (App. C/D)
gamma_sweep.py               sweeps gamma over 256x to test whether the residual
                             looseness is perturbative (Sec. IV); writes
                             gamma_sweep_results.json
fig1.py                      renders Figure 1 from the tabulated values
e1_flux_decomposition.py     (v1.3) flux-identity prediction of every Table III cell from
                             the stabilizer's own cascaded return channel; recovery
                             shortfall and chain loss (Table II, Sec. III.A)
e1c_leff.py                  (v1.3) exact 4x4 leading-order generator L_eff; its slow
                             mode vs the full-Liouvillian spectral rate (Sec. III.A)
e2_gkpq_convergence.py       (v1.3) GKP-q self-rate converged in Fock cutoff N=80..110
e3_block_identities.py       (v1.3) machine check of the block identities, drain identity
                             and exact trace preservation of R used in Supplement S1-S5
verify_nogo.py               (v1.3) tuned-GKP common zero + winding number; cat
                             quadratic-floor scaling fit (Sec. VI, Supplement S6)
verify_nogo2.py              (v1.3) two-tone sine meter zeros on the cat leg basis (Sec. VI)
data/                        saved outputs (JSON) — regenerate with the commands below
```

## Reproduce
```
# Analytic-lemma checks (fast, no eigensolve):
python proof_gkp.py                        # GKP tuned-zero: C(Δ), Δm, κ, Λ01 closed forms vs numerics
                                           #   writes proof_gkp_results.json in the CURRENT directory;
                                           #   compare against data/proof_gkp_results.json
python lemma_A_validation.py               # Assumption-A remainder O(1/g), flux identity, g-sweep
                                           #   writes data/lemma_A_validation.json
python nogo_sweep.py data/nogo_results.json   # menu no-go sweep + floor fits (Sec. VI)
                                           #   the output path is the first argument; with no
                                           #   argument it writes nogo_results.json in the
                                           #   current directory instead

# Table II (tab:bound): all twelve d=2 (code, meter) cells, each checked against
# the printed value as it is produced. ~20 min (dense, N=80 for the GKP rows):
python -u autocorr_extract_d3fix.py table2   > data/table2_cells.txt

# The gamma sweep behind Sec. IV: twelve cells x five gammas, 5e-2 down to
# 1.95e-4. Hours, dense. Writes gamma_sweep_results.json in the current dir;
# the shipped copy is data/gamma_sweep_results.json:
python -u gamma_sweep.py

# Atlas self-rates (dense Liouvillian eigensolves; minutes per cell):
python autocorr_extract_d3fix.py cat3        # cat d=3  (Table I)
python autocorr_extract_d3fix.py cat4        # cat d=4  (Table I)
python autocorr_extract_d3fix.py gkp3 90     # GKP d=3 at Fock cutoff N=90 (Table I)

# Conjugate rates -- the Gamma_conj column of Table I, all rows:
python perception_step3.py       > data/conjugate_rates.txt

# GKP lattice aspect sweep -- Sec. V trade-off curve and the Fig. 1b lattice line:
python -u round10_stageA.py      > data/lattice_sweep_results.txt

# If that sweep dies at r = 1.18 with
#   RuntimeError: SUPERLU_MALLOC fails for buf in intCalloc()
# it is out of memory, not wrong. r > 1 squeezes the q-quadrature into tighter
# combs, so at fixed N = 84 the sparse shift-invert needs more than SuperLU can
# allocate once earlier factorizations have fragmented the heap (~2 GB/process).
# The remaining points complete in fresh processes, same code path, same
# constants:
python -u one_r.py 1.18
python -u one_r.py 1.35

# The figure:
python fig1.py                      # -> readability_fig1.pdf

# v1.3 additions (revision of 2026-10-06). Each writes its JSON next to the
# script; shipped copies are in data/.
python -u e1_flux_decomposition.py all     # ~1 min; run before e1c
python -u e1c_leff.py                      # seconds; reads data/gamma_sweep_results.json
python -u e2_gkpq_convergence.py           # N = 80 90 100 110, ~4 min per N
python -u e3_block_identities.py           # seconds; prints ALL PASS
python -u verify_nogo.py                   # minutes; reads data/nogo_results.json
python -u verify_nogo2.py                  # minutes
```

Table numbering changed in the 2026-10-06 revision: the verification grid
(`tab:bound`, called "Table II" elsewhere in this README) is now **Table III**,
and the new flux-identity table is Table II.

## Reproduction status

Re-run from a clean clone on 2026-08-06 with Python 3.13, numpy 2.2.6,
scipy 1.17.1, qutip 5.3.0. Three scripts reproduce the shipped outputs
**bit-for-bit** — every numeric field identical, worst relative difference
exactly zero:

| script | fields compared | result |
|---|---|---|
| `proof_gkp.py` | 152 | identical |
| `lemma_A_validation.py` | 39 | identical (4/4 checks pass) |
| `nogo_sweep.py` | 613 | identical |

The dense-eigensolve paths (`autocorr_extract_d3fix.py`, `perception_step3.py`,
`round10_stageA.py`, `round10_stageB.py`) were not re-run in that pass; they take
minutes to hours per cell and `round10_stageA.py` has the memory caveat noted
above.

### v1.3 reproduction check (2026-10-06)

All six v1.3 scripts were run from a clean copy of this repository (Python 3.13,
qutip 5.3.0). `e1_flux_decomposition.py` (300 numeric fields), `e1c_leff.py`
(132) and `e2_gkpq_convergence.py` at N = 80 (47) reproduce the shipped outputs
bit-for-bit. `e3_block_identities.py` prints ALL PASS. `verify_nogo.py` and
`verify_nogo2.py` reproduce every quantity the paper quotes (zero locations,
winding numbers, badness floors, fit exponents); their remaining differences
from a July 2026 run are at machine precision (~1e-16) or are equally optimal
parameters returned by a local optimizer in the V-floored meter search.

## Notes
- `qutip==5.3.0` is the version used for the paper; other 5.x releases should work
  but were not validated. The bit-identical re-run above used numpy 2.2.6 and
  scipy 1.17.1 — well above the floors in `requirements.txt` — so the results are
  not sensitive to the exact numpy/scipy versions.
- The `paper (spectral triple-run)` reference values printed by
  `autocorr_extract_d3fix.py` are the *superseded* spectral estimates; the paper
  quotes the windowed-autocorrelation values this script computes (they differ by
  20–50% at d>2 because the spectral extractor misidentifies near-degenerate
  logical modes — see the paper's atlas section).
- The number-code protection scores use a biased recovery pump and are therefore
  conservative, as stated in the paper.
- `lemma_A_validation.py` is an independent, self-contained 4-level model. It
  confirms the *structure* of Appendix C/D: the within-code windowed rate equals
  2*gamma*kappa^2 up to the window deficit of the paper's Eq. (A2) (not the
  spectral eigenvalue), the leak-term flux
  identity holds (to ~9% in this simplified model), which-path back-action gives
  no leading self-damage (residual O(gamma/g)), the slope stays bounded as g
  grows (no Zeno), and the block bounds hold. It is a structural check, not a
  high-precision reproduction of the paper's tightest quoted validation figures.

## License
MIT (see LICENSE).

## Corrections

- **Revision of 2026-10-06 (v1.3).** No shipped number changed. The scripts
  above were added so that the manuscript's new Sec. III.A, Table II, the GKP-q
  convergence statement and Supplement S6 can be reproduced. Two of them were
  developed against a different directory layout; their input paths were
  changed to `data/` for this release, with no change to any computation.

- **Normalization fix, 2026-08-20.** `autocorr_rates_fixed` previously divided
  the windowed slope by `C(0)` rather than `C(t)`, returning the raw derivative
  instead of the logarithmic derivative that `Gamma_self` is defined as. The
  result ran low by `exp(-Gamma t)`: negligible wherever `Gamma*t << 1`, but
  9.5% at cat/parity, where it put the measured rate (0.0904) *below* the bound
  (0.0999) and made the paper's saturation cell look like a violation. Verified
  against the closed form of Eq. (A2): with `n_z ~ 0` the log-derivative must be
  flat at `2*gamma*kappa^2` across the window, and after the fix it is
  (9.99474e-2 -> 9.99254e-2 over t = 0.5..4, against an identity value of
  9.9940e-2), where before it fell 9.512e-2 -> 6.703e-2. **If you are comparing
  against results generated before this date, the fast cells moved.**

- **Two Table II values moved with it.** Only four cells are fast enough for the
  bug to bite (`Gamma*t > 0.05`). Of those, cat/parity and GKP/mod-p had been
  computed by an independent route and were already right; GKP-p and GKP-n had
  not. They were corrected in the manuscript from `4.7e-2` to `5.0e-2` and from
  `9.5e-2` to `1.1e-1`. Both *raise* the measured rate, so both relax the bound
  rather than threaten it. `TABLE2_PRINTED` here tracks the corrected values.

- **`data/gamma_sweep_results.json` predates that correction.** It is the
  genuine 2026-08-19 run and is shipped unedited. Its measured fields (`R`,
  `eta`, rates) are unaffected, but the `published` reference field on the two
  anchor rows still carries `0.047` and `0.095`. `gamma_sweep.py` itself has
  been updated, so a re-run reports against the corrected values.

- **GKP-q is the softest number in Table II.** Its windows are still falling at
  `t = 4`, so `stable()` returns the last one. The printed `3.1e-3` agrees with
  the spectral value to 0.7%, and the "loose by 316x" claim rests on it.

- **`code["logops"]` is ordered `[X, Y, Z]` for the cat codes but `[X, Z, X*Z]`
  for GKP**, so an index rule inferred from `len(logops)` silently reads the
  wrong logical operator for one of them. Carry the index per code.
