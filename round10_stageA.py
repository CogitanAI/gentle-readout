"""Round 10 Stage A: GKP-lattice perception frontier (rectangular deformation).

De-risk stage — map the (protection,
perception) plane over GKP lattice ASPECT RATIO r, find whether an off-square
lattice reads better while protected. Rectangular (shear=0) keeps the logical
q/p labeling clean; r=1 is square GKP (the round-8/9 baseline). All points are
GKP-family (Gaussian-related to square) so a winner is a DESIGN result ("best-
readable GKP"), not a novel code — this stage validates the search pipeline and
calibrates the frontier before the novel Stage B (drain-parametrized).

Lattice: generators g1 = eta*r*q, g2 = (eta/r)*p, eta=sqrt(2*pi*d), det=eta^2=
2*pi*d fixed (so d and ~area held). r>1 squeezes the q-quadrature (tighter
position combs, looser momentum). Finite-energy four-jump stabilizer
(Sellem-Rouchon, as gkp_qudit2). Logical q-spacing = sqrt(2pi/d)/r.

Score per r: P_prot = -log10(worst logical rate, mixed loss+dephasing);
P_perc = best QND readout P_QND = Perceive/self_disturb over a lattice-adapted
observable menu. Pareto vs r=1.

Run:  python -u round10_stageA.py > data/lattice_sweep_results.txt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import qutip as qt
import scipy.linalg

from perception_step3 import induced_rates
from perception_frontier import perceive

EPS = 0.22
N = 84
D = 2
GAMMA, K1, KPHI, KS = 0.05, 0.02, 0.005, 1.0
R_GRID = (0.75, 0.85, 1.0, 1.18, 1.35)


def quads(N):
    a = qt.destroy(N)
    return a, (a + a.dag()) / np.sqrt(2.0), -1j * (a - a.dag()) / np.sqrt(2.0)


def rect_jumps(N, d, r, eps):
    a, q, p = quads(N)
    eta = np.sqrt(2 * np.pi * d)
    Rq = np.cosh(eps) * q + 1j * np.sinh(eps) * p
    Rp = np.cosh(eps) * p - 1j * np.sinh(eps) * q
    g1, g2 = eta * r * Rq, (eta / r) * Rp
    I = qt.qeye(N)
    return [(1j * g1).expm() - I, (-1j * g1).expm() - I,
            (1j * g2).expm() - I, (-1j * g2).expm() - I]


def hermite_psi(n_max, x):
    psi = np.zeros(n_max)
    psi[0] = np.pi**-0.25 * np.exp(-x * x / 2.0)
    if n_max > 1:
        psi[1] = np.sqrt(2.0) * x * psi[0]
    for n in range(1, n_max - 1):
        psi[n + 1] = (np.sqrt(2.0 / (n + 1)) * x * psi[n]
                      - np.sqrt(n / (n + 1.0)) * psi[n - 1])
    return psi


def rect_ansatz(N, d, mu, r, s_max=16):
    spacing = np.sqrt(2 * np.pi / d) / r          # logical q-spacing
    amp = np.zeros(N)
    for s in range(-s_max, s_max + 1):
        amp += hermite_psi(N, spacing * (d * s + mu))
    return qt.Qobj((np.exp(-EPS * np.arange(N)) * amp).reshape(N, 1)).unit()


def codewords(N, d, r, V):
    W = sum(v.dag() * v for v in V)
    evals, evecs = scipy.linalg.eigh(W.full())
    kernel = [qt.Qobj(evecs[:, i].reshape(N, 1)) for i in range(d)]
    Pker = sum(k * k.dag() for k in kernel)
    ws = []
    for mu in range(d):
        w = Pker * rect_ansatz(N, d, mu, r)
        for prev in ws:
            w = w - prev * prev.overlap(w)
        ws.append(w.unit())
    return ws, evals


def adapted_observables(N, d, r):
    a, q, p = quads(N)
    n = qt.num(N)
    parity = (1j * np.pi * n).expm()
    kq = 2 * np.pi / (d * (np.sqrt(2 * np.pi / d) / r))    # matched to q-spacing
    kp = 2 * np.pi / (d * (np.sqrt(2 * np.pi / d) * r))    # p-spacing scales as r
    modq = 0.5 * ((1j * kq * q).expm() + (-1j * kq * q).expm())
    modp = 0.5 * ((1j * kp * p).expm() + (-1j * kp * p).expm())
    return {"n": n, "q": q, "p": p, "parity": parity,
            "mod-q": modq, "mod-p": modp}


def dft(ws):
    d = len(ws)
    om = np.exp(2j * np.pi / d)
    return [sum((om ** (k * m)) * ws[m] for m in range(d)).unit()
            for k in range(d)]


def score(N, d, r):
    a, q, p = quads(N)
    n = qt.num(N)
    V = rect_jumps(N, d, r, EPS)
    ws, evals = codewords(N, d, r, V)
    kernel_ok = evals[d - 1] < 1e-5 and evals[d] > 1e-3
    res = max((v * w).norm() for v in V for w in ws)
    nb = float(np.mean([qt.expect(n, w) for w in ws]))
    if not (kernel_ok and res < 1e-3):
        return None, nb, evals[d - 1], evals[d], res
    om = np.exp(2j * np.pi / d)
    ZL = sum(om**m * ws[m] * ws[m].dag() for m in range(d))
    XL = sum(ws[(m + 1) % d] * ws[m].dag() for m in range(d))
    Pcode = sum(w * w.dag() for w in ws)
    stab = [np.sqrt(KS) * v for v in V]

    # protection: worst logical under mixed noise
    wprot, _ = _worst(stab + [np.sqrt(K1) * a, np.sqrt(KPHI) * n], [XL, ZL])
    P_prot = -np.log10(max(wprot, 1e-12))

    # perception: best QND read over the menu (self-disturb split)
    zst, xst = ws, dft(ws)
    best = (0.0, None)
    for nm, M in adapted_observables(N, d, r).items():
        pz = perceive(M, zst)
        rZ, rX = induced_rates(stab + [np.sqrt(GAMMA) * M], [ZL, XL])
        pq = pz * GAMMA / rZ if rZ > 1e-9 else 0.0
        if pq > best[0]:
            best = (pq, nm)
    return (P_prot, best[0], best[1], wprot), nb, evals[d - 1], evals[d], res


def _worst(c_ops, logops):
    rates = induced_rates(c_ops, logops, rate_ceiling=0.5)
    return max(rates), rates


def main():
    print(f"ROUND 10 STAGE A — GKP lattice perception frontier  "
          f"d={D} N={N} eps={EPS}\n", flush=True)
    print(f"{'r':>6s} {'n_bar':>7s} {'ker_d-1':>9s} {'gap':>7s} "
          f"{'P_prot':>7s} {'P_perc':>8s} {'via':>7s} {'worst':>10s}",
          flush=True)
    rows = []
    for r in R_GRID:
        out, nb, k0, k1, res = score(N, D, r)
        if out is None:
            print(f"{r:6.2f} {nb:7.2f} {k0:9.1e} {k1:7.1e}   "
                  f"CARD FAIL (res={res:.1e}) — skip", flush=True)
            continue
        P_prot, P_perc, via, worst = out
        rows.append((r, nb, P_prot, P_perc, via))
        star = "  <-- square baseline" if abs(r - 1.0) < 1e-9 else ""
        print(f"{r:6.2f} {nb:7.2f} {k0:9.1e} {k1:7.1f} {P_prot:7.2f} "
              f"{P_perc:8.2f} {via:>7s} {worst:10.2e}{star}", flush=True)

    base = next((x for x in rows if abs(x[0] - 1.0) < 1e-9), None)
    if base and len(rows) > 1:
        print(f"\nPareto vs square (P_prot={base[2]:.2f}, P_perc={base[3]:.2f}):",
              flush=True)
        for r, nb, pp, pc, via in rows:
            if abs(r - 1.0) < 1e-9:
                continue
            dom = "DOMINATES" if pp >= base[2] and pc > base[3] else \
                  ("dominated" if pp <= base[2] and pc <= base[3] else "tradeoff")
            print(f"  r={r:.2f}: dProt {pp-base[2]:+.2f} dPerc {pc-base[3]:+.2f}"
                  f"  -> {dom}", flush=True)
    print("\nReading: any r with DOMINATES = off-square GKP reads better while "
          "protected (design result + pipeline validated -> Stage B). All "
          "'tradeoff'/'dominated' = square is Pareto-optimal in the rectangular "
          "family (clean local H0). Verify any dominator dense + at higher N.",
          flush=True)


if __name__ == "__main__":
    main()
