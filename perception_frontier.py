"""The readability frontier: cats vs GKP.

For each encoding, scan P(M,O) = Perceive(M,O)/Disturb(M) over the observable
menu, for reading logical Z and logical X. GKP admits a readout observable
(a modular quadrature) with high P that is also near-orthogonal to the
environment's coupling (loss/dephasing), so it can be read while protected,
where cats cannot.

Functionals (validated in perception_ruler.py):
  Perceive(M,O) = sum over O-eigenstate pairs |<i|M|i> - <j|M|j>|^2   (geometric)
  Disturb(M)    = worst logical rate under stabilizer + gamma*D[M]  (weak gamma)
  P(M,O)        = Perceive(M,O) * gamma / Disturb(M)   (gamma-independent)

Observable menu (Hermitian): n, q, p, parity, modular-q, modular-p.
  modular-q = cos(k q), k = 2pi/(d * spacing_L), spacing_L = sqrt(2pi/d):
  distinguishes the d logical cosets of a LATTICE code, ~blind to cats.

Encodings: d=2 cat, d=4 cat, GKP qubit, GKP qutrit (matched-ish n_bar).
The ENVIRONMENT reference rows (loss D[a], dephasing D[n]) show where the noise
sits: high Disturb, ~zero Perceive. The readout sweet spot = high P AND an
observable the environment does not couple to.

Run:  python perception_frontier.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import qutip as qt

from qudlab.codes import (auto_cutoff, cat_states, code_projector,
                          coherent_legs, logical_paulis_d2, weyl_operators)
from qudlab.codes import _orthonormal_span
from qudlab.loss_correction import worst_logical_rate
from gkp_qudit2 import EPS, gkp_codewords, gkp_jumps

GAMMA, KD, KAPPA1 = 0.05, 1.0, 0.02


def dft_states(z_states):
    """X-eigenstates = discrete Fourier transform of the Z-eigenstate (coset) basis."""
    d = len(z_states)
    om = np.exp(2j * np.pi / d)
    out = []
    for k in range(d):
        v = sum((om ** (k * m)) * z_states[m] for m in range(d)) / np.sqrt(d)
        out.append(v.unit())
    return out


def observables(N, d):
    a = qt.destroy(N)
    n = qt.num(N)
    q = (a + a.dag()) / np.sqrt(2.0)
    p = -1j * (a - a.dag()) / np.sqrt(2.0)
    parity = (1j * np.pi * n).expm()
    spacing_L = np.sqrt(2 * np.pi / d)
    k = 2 * np.pi / (d * spacing_L)
    modq = 0.5 * ((1j * k * q).expm() + (-1j * k * q).expm())   # cos(k q)
    modp = 0.5 * ((1j * k * p).expm() + (-1j * k * p).expm())   # cos(k p)
    return {"n": n, "q": q, "p": p, "parity": parity,
            "mod-q": modq, "mod-p": modp}, a


def perceive(M, states):
    ex = [qt.expect(M, s) for s in states]
    return float(sum(abs(ex[i] - ex[j]) ** 2
                     for i in range(len(ex)) for j in range(i + 1, len(ex))))


def disturb(M, stab_cops, logops, Pcode, rate=GAMMA):
    worst, _, _ = worst_logical_rate(stab_cops + [np.sqrt(rate) * M],
                                     logops, Pcode)
    return worst


def scan(code):
    name, N, d = code["name"], code["N"], code["d"]
    z_st, x_st = code["z_states"], code["x_states"]
    stab, logops, Pcode, a = (code["stab"], code["logops"], code["Pcode"],
                              code["a"])
    obs, _ = observables(N, d)
    print(f"\n=== {name}  (d={d}, N={N}) ===", flush=True)
    print(f"{'observable':>10s} {'Perc(Z)':>10s} {'Perc(X)':>10s} "
          f"{'Disturb':>10s} {'P_Z':>10s} {'P_X':>10s}", flush=True)
    best = {"Z": (0.0, None), "X": (0.0, None)}
    for nm, M in obs.items():
        pz, px = perceive(M, z_st), perceive(M, x_st)
        dst = disturb(M, stab, logops, Pcode)
        Pz = pz * GAMMA / dst if dst > 1e-12 else float('inf')
        Px = px * GAMMA / dst if dst > 1e-12 else float('inf')
        if np.isfinite(Pz) and Pz > best["Z"][0]:
            best["Z"] = (Pz, nm)
        if np.isfinite(Px) and Px > best["X"][0]:
            best["X"] = (Px, nm)
        print(f"{nm:>10s} {pz:10.3e} {px:10.3e} {dst:10.3e} "
              f"{Pz:10.3e} {Px:10.3e}", flush=True)
    # environment reference
    d_loss = disturb(a, stab, logops, Pcode, rate=KAPPA1)
    d_deph = disturb(qt.num(N), stab, logops, Pcode, rate=KAPPA1)
    print(f"{'env:loss':>10s} {'(n/a)':>10s} {'(n/a)':>10s} {d_loss:10.3e}",
          flush=True)
    print(f"{'env:deph':>10s} {'(n/a)':>10s} {'(n/a)':>10s} {d_deph:10.3e}",
          flush=True)
    print(f"  best readout:  Z via {best['Z'][1]} (P={best['Z'][0]:.2e})   "
          f"X via {best['X'][1]} (P={best['X'][0]:.2e})", flush=True)
    return best


def cat_code(alpha, d):
    N = auto_cutoff(alpha, d)
    a = qt.destroy(N)
    z_st = _orthonormal_span(coherent_legs(N, alpha, d))   # legs = Z(clock)
    x_st = cat_states(N, alpha, d)                         # cats = X(shift)
    if d == 2:
        L = logical_paulis_d2(N, alpha)
        logops = [L["X"], L["Y"], L["Z"]]
    else:
        W = weyl_operators(N, alpha, d)
        logops = [W["X"], W["Z"]]
    return {"name": f"cat d={d} a={alpha}", "N": N, "d": d, "a": a,
            "z_states": z_st, "x_states": x_st,
            "stab": [np.sqrt(KD) * (a**d - alpha**d)],
            "logops": logops, "Pcode": code_projector(N, alpha, d)}


def gkp_code(d, N):
    V = gkp_jumps(N, d, EPS)
    ws, evals = gkp_codewords(N, d, EPS, V)
    om = np.exp(2j * np.pi / d)
    Z = sum(om**m * ws[m] * ws[m].dag() for m in range(d))
    X = sum(ws[(m + 1) % d] * ws[m].dag() for m in range(d))
    Pcode = sum(w * w.dag() for w in ws)
    logops = [X, Z, X * Z]
    kernel_ok = evals[d - 1] < 1e-5
    return {"name": f"GKP d={d}", "N": N, "d": d, "a": qt.destroy(N),
            "z_states": ws, "x_states": dft_states(ws),
            "stab": [np.sqrt(KD) * v for v in V],
            "logops": logops, "Pcode": Pcode, "kernel_ok": kernel_ok}


def main():
    print("PERCEPTION FRONTIER  cats vs GKP  (gamma={:.2f})".format(GAMMA),
          flush=True)
    scan(cat_code(2.0, 2))
    scan(cat_code(2.0, 4))
    for d, N in ((2, 80), (3, 90)):
        g = gkp_code(d, N)
        if not g["kernel_ok"]:
            print(f"\n=== GKP d={d}: kernel not clean at N={N}, skipping ===",
                  flush=True)
            continue
        scan(g)
    print("\nReading: compare the CAT best-readout P vs the GKP best-readout P, "
          "AND which observable achieves it. Claim confirmed if GKP reads a "
          "logical op via mod-q/mod-p (an observable the env doesn't couple to) "
          "at P >> the cats' best, i.e. readable-while-protected.", flush=True)


if __name__ == "__main__":
    main()
