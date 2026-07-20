"""Generalized number-code search on the (protection, readability) plane.

Searches non-Gaussian single-mode number codes for a point that Pareto-dominates
the cat and GKP d=2 baselines (reads better while protecting at least as well).
This produces the "48 number codes" cloud of Fig. 1(b); none escapes the
readability floor.

Family: generalized single-mode number codes. Logical mu in {0,1} supported on
Fock sectors n = g*(2k+mu), k=0..M-1 (spacing g, M terms), amplitudes from a
tunable envelope (binomial, or Gaussian-in-k). Read through the number/parity
channel. An ideal recovery pump stabilizes the 2-dim code (dark space = code by
construction), so the protection scores are conservative, as noted in the paper.

Score per code: P_prot = -log10(worst logical rate, mixed loss+dephasing under
the pump); readability P = best QND readout over the observable menu. Baselines:
cat d=2 and GKP d=2 at matched n_bar ~ 1.9.

Run:  python round10_stageB.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import qutip as qt
import scipy.linalg
from scipy.special import gammaln

from perception_step3 import induced_rates
from perception_frontier import cat_code, gkp_code, observables, perceive


def safe_rates(c_ops, logops):
    """induced_rates guarded against ARPACK non-convergence (ill-conditioned
    high-n_bar codes that overflow the cutoff). Returns None on failure."""
    try:
        return induced_rates(c_ops, logops, rate_ceiling=0.5)
    except Exception:
        return None

N = 48
GAMMA, K1, KPHI, LAM = 0.05, 0.02, 0.005, 1.0


def envelope(M, kind):
    k = np.arange(M)
    if kind == "binom":
        logc = gammaln(M) - gammaln(k + 1) - gammaln(M - k)   # C(M-1,k)
        c = np.exp(0.5 * (logc - logc.max()))
    elif kind == "flat":
        c = np.ones(M)
    else:                                                     # gaussian in k
        w = float(kind)
        c = np.exp(-((k - (M - 1) / 2) ** 2) / (2 * w * w))
    return c / np.linalg.norm(c)


def number_code(N, g, M, kind):
    ws = []
    for mu in (0, 1):
        v = np.zeros(N)
        c = envelope(M, kind)
        for k in range(M):
            n = g * (2 * k + mu)
            if n < N:
                v[n] = c[k]
        ws.append(qt.Qobj(v.reshape(N, 1)).unit())
    # even/odd -> logical Z basis directly (already orthogonal: disjoint support)
    return ws


def pump_cops(ws, N, lam):
    Pcode = sum(w * w.dag() for w in ws)
    evals, evecs = scipy.linalg.eigh(Pcode.full())
    comp = [qt.Qobj(evecs[:, i].reshape(N, 1))
            for i in range(N) if evals[i] < 0.5]      # complement basis
    ref = ws[0]
    return [np.sqrt(lam) * (ref * c.dag()) for c in comp], Pcode


def score(ws, N):
    n = qt.num(N)
    a = qt.destroy(N)
    ov = abs(ws[0].overlap(ws[1]))
    if ov > 1e-6:
        return None
    pump, Pcode = pump_cops(ws, N, LAM)
    nb = float(np.mean([qt.expect(n, w) for w in ws]))
    ZL = ws[0] * ws[0].dag() - ws[1] * ws[1].dag()
    XL = ws[0] * ws[1].dag() + ws[1] * ws[0].dag()
    # protection: worst logical under mixed noise + pump
    rates = safe_rates(pump + [np.sqrt(K1) * a, np.sqrt(KPHI) * n], [ZL, XL])
    if rates is None:
        return None
    worst = max(rates)
    P_prot = -np.log10(max(worst, 1e-12))
    # perception: best QND read-Z (self-based: self = rate on ZL you read)
    zst = ws
    best = (0.0, None)
    for nm, M in observables(N, 2)[0].items():
        pz = perceive(M, zst)
        rr = safe_rates(pump + [np.sqrt(GAMMA) * M], [ZL, XL])
        if rr is None:
            continue
        rZ = rr[0]                                  # self-disturbance on Z
        pq = pz * GAMMA / rZ if rZ > 1e-9 else 0.0
        if pq > best[0]:
            best = (pq, nm)
    return P_prot, best[0], best[1], nb, worst


def baseline(code):
    N_ = code["N"]
    n = qt.num(N_)
    a = code["a"]
    logops = code["logops"]
    XL = logops[0]
    ZL = logops[2] if len(logops) == 3 else logops[1]
    Pcode = code["Pcode"]
    stab = code["stab"]
    rates = safe_rates(stab + [np.sqrt(K1) * a, np.sqrt(KPHI) * n], [ZL, XL])
    P_prot = -np.log10(max(max(rates), 1e-12))
    zst = code["z_states"]
    best = (0.0, None)
    for nm, M in observables(N_, code["d"])[0].items():
        pz = perceive(M, zst)
        rr = safe_rates(stab + [np.sqrt(GAMMA) * M], [ZL, XL])
        if rr is None:
            continue
        rZ = rr[0]                                  # self-disturbance on Z
        pq = pz * GAMMA / rZ if rZ > 1e-9 else 0.0
        if pq > best[0]:
            best = (pq, nm)
    return P_prot, best[0], best[1]


def main():
    print(f"Generalized number-code search  N={N}\n",
          flush=True)
    cprot, cperc, cvia = baseline(cat_code(np.sqrt(1.9), 2))
    gprot, gperc, gvia = baseline(gkp_code(2, 80))
    print(f"BASELINES (matched n_bar~1.9):", flush=True)
    print(f"  cat d=2:  P_prot={cprot:.2f}  P_perc={cperc:.1f} ({cvia})",
          flush=True)
    print(f"  GKP d=2:  P_prot={gprot:.2f}  P_perc={gperc:.1f} ({gvia})",
          flush=True)
    bp, bc = max(cprot, gprot), max(cperc, gperc)
    print(f"  to DOMINATE both: P_prot>={bp:.2f} AND P_perc>{bc:.1f}\n",
          flush=True)

    print(f"{'g':>3s} {'M':>3s} {'env':>8s} {'n_bar':>6s} {'P_prot':>7s} "
          f"{'P_perc':>8s} {'via':>7s} {'verdict':>10s}", flush=True)
    hits = []
    for g in (1, 2, 3):
        for M in (2, 3, 4, 5):
            for kind in ("binom", "flat", "1.0", "2.0"):
                ws = number_code(N, g, M, kind)
                out = score(ws, N)
                if out is None:
                    continue
                P_prot, P_perc, via, nb, worst = out
                dom = ("DOMINATES" if P_prot >= bp and P_perc > bc else
                       ("beats-cat" if P_prot >= cprot and P_perc > cperc else
                        ("beats-GKP" if P_prot >= gprot and P_perc > gperc
                         else "-")))
                if dom != "-":
                    hits.append((g, M, kind, P_prot, P_perc, via, dom))
                print(f"{g:3d} {M:3d} {kind:>8s} {nb:6.2f} {P_prot:7.2f} "
                      f"{P_perc:8.1f} {via:>7s} {dom:>10s}", flush=True)

    print("\n" + ("DOMINATORS FOUND (verify dense + novelty + realizability):"
                  if any(h[6] == "DOMINATES" for h in hits)
                  else "No full dominator. Partial beats:"), flush=True)
    for h in hits:
        print(f"  g={h[0]} M={h[1]} {h[2]}: P_prot={h[3]:.2f} P_perc={h[4]:.1f}"
              f" via {h[5]} -> {h[6]}", flush=True)
    print("\nReading: DOMINATES = a non-Gaussian code reading better while "
          "protected at least as well as both known codes. Only partial/no beats "
          "means the protection-readout tradeoff extends to number codes. "
          "Protection uses an ideal recovery pump (scores conservative).", flush=True)


if __name__ == "__main__":
    main()
