"""Validation "ruler" for the readability functionals.

Validates the two functionals (PERCEIVE, DISTURB) against the known limits of
the d=2 cat qubit.

Functionals (measurement rate gamma = 1; it cancels in the ratio):
  PERCEIVE(M, read O) = sum over the two O-eigenstates |i>,|j| of
                        |<i|M|i> - <j|M|j>|^2   (leading Fisher/SNR content of a
                        weak continuous measurement of Hermitian M; = 0 exactly
                        when M cannot distinguish the O-basis states)
  DISTURB(M)          = worst logical rate under the stabilizer + gamma*D[M]
                        (the measurement backaction; validated worst_logical_rate)
  P(M, O)             = PERCEIVE(M,O) / DISTURB(M)  = logical bits read per unit
                        logical damage.

d=2 cat logical bases (codes.py convention, |0_L>=|alpha>,|1_L>=|-alpha>):
  Z-eigenstates = the LEGS {|alpha>, |-alpha>}         (read Z = "which leg")
  X-eigenstates = the CATS {|even>, |odd>}             (read X = "which parity")

RULER (must reproduce, or the functional is wrong):
  R1  parity reads X (cats) but NOT Z (legs):  Perceive(par,X) >> Perceive(par,Z)~0
  R2  number n reads NEITHER logical op (damages without perceiving):
      Perceive(n,Z) ~ 0 AND Perceive(n,X) ~ 0, while Disturb(n) > 0
  R3  a quadrature reads Z (the legs):         Perceive(q,Z) >> 0
  R4  environment channels (loss D[a], dephasing D[n]) have Perceive(logical)~0
      but Disturb>0  -> P ~ 0: the environment damages without perceiving = the
      essence of protection.
  R5  no observable reads a logical op with zero disturbance (info-disturbance):
      every M with Perceive(.,O)>0 has Disturb(M)>0.

Run:  python perception_ruler.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import qutip as qt

from qudlab.codes import (auto_cutoff, cat_states, code_projector,
                          coherent_legs, logical_paulis_d2)
from qudlab.codes import _orthonormal_span
from qudlab.loss_correction import worst_logical_rate

ALPHA, KD = 2.0, 1.0
GAMMA = 0.05                    # WEAK measurement rate: keeps induced logical
#                                rates inside the extractor window so P=Perc/Dist
#                                is clean & gamma-independent (both scale with g).
KAPPA1 = 0.02                   # fixed environmental loss rate (reference row)
TOL = 1e-3                      # "~0" threshold for a Perceive that should vanish
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)


def perceive(M, s_i, s_j):
    """|<i|M|i> - <j|M|j>|^2 for the two eigenstates of the read-out operator."""
    di = qt.expect(M, s_i)
    dj = qt.expect(M, s_j)
    return float(abs(di - dj) ** 2)


def disturb(M_jump, stab, logops, Pcode, rate=GAMMA):
    """worst logical rate under the stabilizer + the measurement/loss channel."""
    c_ops = [np.sqrt(KD) * stab, np.sqrt(rate) * M_jump]
    worst, _, _ = worst_logical_rate(c_ops, logops, Pcode)
    return worst


def main():
    N = auto_cutoff(ALPHA, 2)
    a = qt.destroy(N)
    n = qt.num(N)
    q = (a + a.dag()) / np.sqrt(2.0)
    p = -1j * (a - a.dag()) / np.sqrt(2.0)
    parity = (1j * np.pi * n).expm()
    stab = a * a - ALPHA**2

    L = logical_paulis_d2(N, ALPHA)
    logops = [L["X"], L["Y"], L["Z"]]
    Pcode = code_projector(N, ALPHA, 2)

    e0, e1 = _orthonormal_span(coherent_legs(N, ALPHA, 2))    # Z-eigenstates
    cats = cat_states(N, ALPHA, 2)                            # X-eigenstates
    even, odd = cats[0], cats[1]

    print(f"PERCEPTION-DUALITY RULER  d=2 cat  alpha={ALPHA}  N={N}\n", flush=True)
    print(f"{'observable':>12s} {'Perceive(Z=legs)':>17s} "
          f"{'Perceive(X=cats)':>17s} {'Disturb':>10s} {'P_Z':>10s} {'P_X':>10s}",
          flush=True)

    hermitian = {"n": n, "q": q, "p": p, "parity": parity}
    vals = {}
    for name, M in hermitian.items():
        pz = perceive(M, e0, e1)
        px = perceive(M, even, odd)
        dst = disturb(M, stab, logops, Pcode)
        vals[name] = (pz, px, dst)
        # P = bits read per unit logical damage; gamma-normalized (Disturb ~ gamma)
        Pz = pz * GAMMA / dst if dst > 1e-12 else float('inf')
        Px = px * GAMMA / dst if dst > 1e-12 else float('inf')
        print(f"{name:>12s} {pz:17.4e} {px:17.4e} {dst:10.4e} "
              f"{Pz:10.3e} {Px:10.3e}", flush=True)

    # loss channel (non-Hermitian jump): Disturb only, Perceive N/A
    dst_loss = disturb(a, stab, logops, Pcode, rate=KAPPA1)
    print(f"{'loss D[a]':>12s} {'(n/a)':>17s} {'(n/a)':>17s} "
          f"{dst_loss:10.4e} {'-':>10s} {'-':>10s}", flush=True)

    # ------------------------------------------------------------------ ruler
    print("\nRULER:", flush=True)
    pz_par, px_par, d_par = vals["parity"]
    pz_n, px_n, d_n = vals["n"]
    pz_q, px_q, d_q = vals["q"]
    check("R1 parity reads X(cats) not Z(legs)",
          px_par > 0.5 and pz_par < TOL,
          f"P(par,X)={px_par:.3f}  P(par,Z)={pz_par:.1e}")
    check("R2 number n reads NEITHER logical op (damages w/o perceiving)",
          pz_n < TOL and px_n < TOL and d_n > 1e-6,
          f"P(n,Z)={pz_n:.1e}  P(n,X)={px_n:.1e}  Disturb={d_n:.2e}")
    check("R3 quadrature q reads Z(legs)",
          pz_q > 0.5, f"P(q,Z)={pz_q:.3f}")
    check("R4 environment damages without perceiving (P~0)",
          (pz_n * GAMMA / d_n if d_n > 0 else 0) < TOL and dst_loss > 1e-6,
          f"P_Z(n)={pz_n * GAMMA / d_n if d_n>0 else 0:.1e}  "
          f"Disturb(loss)={dst_loss:.2e}")
    # R5 info-disturbance: any observable that reads a logical op must disturb
    reads = [(nm, pz, px, d) for nm, (pz, px, d) in vals.items()
             if max(pz, px) > 0.5]
    check("R5 no zero-disturbance logical readout (info-disturbance)",
          all(d > 1e-9 for _, _, _, d in reads),
          f"readers={[nm for nm, *_ in reads]}, min Disturb="
          f"{min((d for *_, d in reads), default=float('nan')):.2e}")

    print(f"\nRULER: {'PASS -> functionals trustworthy, proceed to frontier' if OK else 'FAIL -> fix before trusting any number'}",
          flush=True)


if __name__ == "__main__":
    main()
