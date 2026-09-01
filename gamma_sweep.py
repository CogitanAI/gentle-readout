"""Gamma sweep of the readability bound: is the out-of-regime looseness perturbative?

Table II of readability.tex verifies Gamma_self >= gamma*max(2 kappa^2, eps_KL^2/2)
at gamma = 0.05. Six of the twelve cells sit OUTSIDE hypothesis (G3),
eta = gamma m^2 / g <= 1/32, because q, p and n are unbounded meters. An earlier
draft attributed their looseness to uncontrolled O(gamma^2/g) terms. That was an
assertion with nothing behind it, and it is testable -- this script is the test.
It failed: R is flat, so the explanation was withdrawn and the published text
reports the measured result instead.

Both bound terms are exactly linear in gamma (kappa and Lambda are gamma-free),
and eta is exactly linear in gamma. So sweeping gamma down:
  - drives every cell inside (G3) at small enough gamma (GKP-n, the worst at
    eta = 3.9, enters below gamma ~ 4e-4), and
  - if the manuscript is right, drives R = Gamma_self / bound DOWN toward its
    leading-order value.
R flat => the looseness is structural, not perturbative. That is what the sweep
found: R moves by at most 16% across a 256x range in gamma, and at two cells it
increases.

LOGICAL OPERATOR (bug found 2026-08-19): the two codes order logops differently
-- cat d=2 is [X, Y, Z], GKP is [X, Z, X*Z] -- and theorem_refit.py selects
index 2 for both, which silently reads X*Z instead of Z for every GKP cell.
Table II's printed GKP values match the Z computation (by an independent dense audit),
so the published table is correct and the on-disk generator is not. The index is
carried per code here rather than inferred from len(logops).

ESTIMATOR: full dense spectrum, top-overlap mode selection -- the procedure
the audit path used, and the one Table II's printed values agree with.
NOT induced_rates(): that k=30 sparse shift-invert mis-identifies modes in
overlap-tie regimes at N=80 and today returns 5.00e-2 for GKP-q against a dense
3.12e-3 (16x) and 2.00e-1 for GKP-n against a dense 9.50e-2 (2.1x). Dense costs
2.5 min/cell at N=80 and ~10 s at N=48, so there is no reason to use anything else.

The gamma = 0.05 row is the anchor and must reproduce Table II; it is reported
against the printed values on every line.

Run:  python -u gamma_sweep.py     # writes gamma_sweep_results.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import qutip as qt
import scipy.linalg

from perception_frontier import cat_code, gkp_code, observables

ANCHOR = 0.05
GAMMAS = [0.05, 1.25e-2, 3.125e-3, 7.8125e-4, 1.953125e-4]
METERS = ["parity", "mod-q", "mod-p", "q", "p", "n"]
CEIL = 0.5
BOUND_MIN = 1e-30   # below this the bound is numerically zero (GKP-parity is blind)

# Table II as printed, for the anchor check (GKP-p and GKP-n carry the
# 2026-08-20 corrections; see the note in README.md)
PUBLISHED = {
    ("cat d=2", "parity"): 1.0e-1, ("cat d=2", "mod-q"): 1.7e-3,
    ("cat d=2", "mod-p"): 9.4e-3, ("cat d=2", "q"): 1.2e-3,
    ("cat d=2", "p"): 1.5e-3, ("cat d=2", "n"): 4.8e-3,
    ("GKP d=2", "parity"): 1.4e-8, ("GKP d=2", "mod-q"): 1.7e-3,
    ("GKP d=2", "mod-p"): 7.4e-2, ("GKP d=2", "q"): 3.1e-3,
    ("GKP d=2", "p"): 5.0e-2, ("GKP d=2", "n"): 1.1e-1}
# eta at the anchor (Table II); exactly linear in gamma
ETA0 = {("cat d=2", "parity"): 0.010, ("cat d=2", "mod-q"): 0.007,
        ("cat d=2", "mod-p"): 0.007, ("cat d=2", "q"): 0.089,
        ("cat d=2", "p"): 0.089, ("cat d=2", "n"): 0.51,
        ("GKP d=2", "parity"): 0.015, ("GKP d=2", "mod-q"): 0.013,
        ("GKP d=2", "mod-p"): 0.013, ("GKP d=2", "q"): 0.36,
        ("GKP d=2", "p"): 0.36, ("GKP d=2", "n"): 3.9}


def dense_rate(c_ops, O, floor):
    """Top-overlap logical mode from the full dense Liouvillian spectrum.
    Returns (rate, top3) so overlap ties stay visible in the record."""
    Lsuper = qt.liouvillian(0 * c_ops[0], c_ops)
    vals, vecs = scipy.linalg.eig(Lsuper.full())
    rates = -vals.real
    of = O.full().reshape(-1, order="F").conj()
    ovs = np.abs(of @ vecs) / (np.linalg.norm(vecs, axis=0) + 1e-15)
    mask = (rates > floor) & (rates < CEIL)
    idx = np.argsort(np.where(mask, ovs, -1.0))[::-1][:3]
    top = [(float(ovs[i]), float(rates[i])) for i in idx]
    return (top[0][1] if top[0][0] > 0.1 else 0.0), top


def lam_split(M, ws, Pcode):
    """kappa^2 and the KL off-diagonal weight -- both gamma-independent."""
    N = M.shape[0]
    Pp = qt.qeye(N) - Pcode
    L = np.array([[complex(ws[i].overlap(M * (Pp * (M * ws[j]))))
                   for j in range(2)] for i in range(2)])
    return (abs(complex(ws[0].overlap(M * ws[1]))) ** 2,
            abs(L[0, 1]) ** 2, float(np.real(np.trace(L))))


def main():
    out = []
    # (label, code, index of logical Z in code["logops"])
    codes = [("cat d=2", cat_code(np.sqrt(1.9), 2), 2),
             ("GKP d=2", gkp_code(2, 80), 1)]
    print(f"GAMMA SWEEP (dense)  gammas={GAMMAS}", flush=True)
    print(f"{'code':<8s}{'meter':>7s}{'gamma':>11s}{'eta':>8s}{'G3':>4s}"
          f"{'Gamma_self':>12s}{'bound':>11s}{'R':>9s}{'tie':>6s}{'min':>6s}",
          flush=True)
    for short, code, zli in codes:
        N = code["N"]
        ZL = code["logops"][zli]
        ws, Pcode, stab = code["z_states"], code["Pcode"], code["stab"]
        obs, _ = observables(N, 2)
        for meter in METERS:
            M = obs[meter]
            kappa2, off2, trL = lam_split(M, ws, Pcode)
            for g in GAMMAS:
                floor = 1e-12 * (g / ANCHOR)
                t0 = time.time()
                rate, top = dense_rate(stab + [np.sqrt(g) * M], ZL, floor)
                dt = time.time() - t0
                b1, b2 = 2 * g * kappa2, (g * off2 / (2 * trL) if trL > 1e-12 else 0.0)
                bound = max(b1, b2)
                R = rate / bound if bound > BOUND_MIN else float("nan")
                eta = ETA0[(short, meter)] * (g / ANCHOR)
                # overlap tie ratio: 2nd-best / best. ~1 means the pick is fragile.
                tie = top[1][0] / top[0][0] if top[0][0] > 0 else float("nan")
                rec = dict(code=short, meter=meter, gamma=g, eta=eta,
                           in_G3=bool(eta <= 1 / 32), gamma_self=rate,
                           b_kappa=b1, b_kl=b2, bound=bound, R=R,
                           tie=tie, top3=top, secs=dt)
                if g == ANCHOR:
                    pub = PUBLISHED[(short, meter)]
                    rec["published"] = pub
                    rec["anchor_ratio"] = rate / pub if pub else float("nan")
                out.append(rec)
                mark = "in" if eta <= 1 / 32 else "OUT"
                extra = (f"   anchor vs published {PUBLISHED[(short,meter)]:.1e}"
                         f" -> {rec['anchor_ratio']:.2f}x" if g == ANCHOR else "")
                rs = f"{R:>9.2f}" if np.isfinite(R) else f"{'blind':>9s}"
                print(f"{short:<8s}{meter:>7s}{g:>11.3e}{eta:>8.4f}{mark:>4s}"
                      f"{rate:>12.3e}{bound:>11.3e}{rs}{tie:>6.2f}"
                      f"{dt/60:>6.1f}{extra}", flush=True)
                json.dump(out, open("gamma_sweep_results.json", "w"), indent=1)
    print("\nReading: per cell, compare R at gamma=0.05 with R at gamma=1.95e-4. "
          "R falling toward 1 => the looseness was the O(gamma^2/g) remainder, as "
          "the manuscript claims. R flat => structural, and that claim goes.",
          flush=True)


if __name__ == "__main__":
    main()
