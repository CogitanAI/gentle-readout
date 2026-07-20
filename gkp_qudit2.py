"""GKP-qudit construction on the four-jump dissipative Lindblad scheme.

Scheme (Sellem-Campagne-Ibarcq-Mirrahimi-Sarlette-Rouchon, arXiv:2203.16836;
conventions [q,p]=i, q=(a+a†)/sqrt2, eps = Delta^2, E = e^{-eps n}):

    R = cosh(eps) q + i sinh(eps) p        (= E q E^{-1})
    S = cosh(eps) p - i sinh(eps) q        (= E p E^{-1})
    eta_d = sqrt(2 pi d)                    (qubit: 2 sqrt(pi))
    V_{1..4} = e^{+i eta R} - 1, e^{-i eta R} - 1,
               e^{+i eta S} - 1, e^{-i eta S} - 1     (all rate GAMMA)

Exact facts (paper, d=2; and for d>2 --- commutation phase
e^{-i eta^2} = e^{-2 pi i d} = 1 for all integer d): the four V_k commute,
their joint kernel is exactly the d-dim finite-energy code space, the
Liouvillian kernel is exactly d^2-dim, gap ~ eta^2 eps GAMMA. The d>2 spectra
are computed numerically here.

Numerical rules (from the recipe): never form E^{-1}; build e^{i eta R} by
.expm() of the tridiagonal generator. Codewords = kernel of the Hermitian
W = sum V†V, labeled by projecting position-comb ansatz states onto ker(W).

Bench: worst logical (Weyl set) under loss k1=0.02 (< eps/5 regime) and
dephasing kphi=0.005, d=2 vs d=3 at the SAME eps (same n_bar) — the qudit
penalty ratio, against the <=3x practicality bar and the pair-cat's 7-14x.

Sizes: card + W checks at N_CHK=125; Liouvillian spectra at N=90 (dense);
mesolve time-domain cross-check for the two mixed-noise headline numbers.

Run:  python gkp_qudit2.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import qutip as qt
import scipy.linalg
from scipy.special import gammaln

N_SPEC = {2: 95, 3: 105}  # d=3 jumps kick harder -> needs more Fock headroom
N_CHK = 125
EPS = 0.22                # Delta~0.47; respects the N >~ 20/eps rule (d=2)
GAMMA, K1, KPHI = 1.0, 0.02, 0.005
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)


def quad_ops(N):
    a = qt.destroy(N)
    return (a + a.dag()) / np.sqrt(2.0), -1j * (a - a.dag()) / np.sqrt(2.0)


def gkp_jumps(N, d, eps):
    q, p = quad_ops(N)
    R = np.cosh(eps) * q + 1j * np.sinh(eps) * p
    S = np.cosh(eps) * p - 1j * np.sinh(eps) * q
    eta = np.sqrt(2.0 * np.pi * d)
    I = qt.qeye(N)
    V = [(1j * eta * R).expm() - I, (-1j * eta * R).expm() - I,
         (1j * eta * S).expm() - I, (-1j * eta * S).expm() - I]
    return V


def hermite_psi(n_max, x):
    psi = np.zeros(n_max)
    psi[0] = np.pi**-0.25 * np.exp(-x * x / 2.0)
    if n_max > 1:
        psi[1] = np.sqrt(2.0) * x * psi[0]
    for n in range(1, n_max - 1):
        psi[n + 1] = (np.sqrt(2.0 / (n + 1)) * x * psi[n]
                      - np.sqrt(n / (n + 1.0)) * psi[n - 1])
    return psi


def ansatz_comb(N, d, mu, eps, s_max=14):
    """Rough enveloped position comb (labeling only; O(eps) accurate)."""
    spacing = np.sqrt(2.0 * np.pi / d)
    amp = np.zeros(N)
    for s in range(-s_max, s_max + 1):
        amp += hermite_psi(N, spacing * (d * s + mu))
    v = np.exp(-eps * np.arange(N)) * amp
    return qt.Qobj(v.reshape(N, 1)).unit()


def gkp_codewords(N, d, eps, V):
    """Exact kernel of W = sum V†V, physically labeled via comb projection."""
    W = sum(v.dag() * v for v in V)
    evals, evecs = scipy.linalg.eigh(W.full())
    kernel = [qt.Qobj(evecs[:, i].reshape(N, 1)) for i in range(d)]
    Pker = sum(k * k.dag() for k in kernel)
    ws = []
    for mu in range(d):
        w = Pker * ansatz_comb(N, d, mu, eps)
        for prev in ws:
            w = w - prev * prev.overlap(w)
        ws.append(w.unit())
    return ws, evals


def weyl_set(ws):
    d = len(ws)
    om = np.exp(2j * np.pi / d)
    Z = sum(om**mu * ws[mu] * ws[mu].dag() for mu in range(d))
    X = sum(ws[(mu + 1) % d] * ws[mu].dag() for mu in range(d))
    return [(X**j) * (Z**k) for j in range(d) for k in range(d)
            if (j, k) != (0, 0)]


def dense_logical_fast(c_ops, logops, ceiling):
    Lsuper = qt.liouvillian(0 * c_ops[0], c_ops)
    vals, vecs = scipy.linalg.eig(Lsuper.full())
    rates = -vals.real
    mask = (rates > 1e-9) & (rates < ceiling)
    worst = 0.0
    for O in logops:
        of = O.full().reshape(-1, order="F").conj()
        ovs = np.abs(of @ vecs)
        ovs = np.where(mask, ovs, -1.0)
        i = int(np.argmax(ovs))
        worst = max(worst, rates[i] if ovs[i] > 0.1 else 0.0)
    return worst


def main():
    num_chk = qt.num(N_CHK)
    print(f"GKP QUDIT BENCH v2 (four-jump Lindblad)  eps={EPS} "
          f"N_chk={N_CHK} N_spec={N_SPEC}\n", flush=True)

    # ---------------- card at N_CHK ----------------
    words_chk = {}
    for d in (2, 3):
        V = gkp_jumps(N_CHK, d, EPS)
        ws, evals = gkp_codewords(N_CHK, d, EPS, V)
        words_chk[d] = ws
        kernel_ratio = evals[d] / max(evals[d - 1], 1e-300)
        res = max((v * w).norm() for v in V for w in ws)
        nbar = float(np.mean([qt.expect(num_chk, w) for w in ws]))
        comb_ov = min(abs(ws[mu].overlap(ansatz_comb(N_CHK, d, mu, EPS)))
                      for mu in range(d))
        check(f"d={d}: W kernel is exactly {d}-dim",
              evals[d - 1] < 1e-6 and kernel_ratio > 1e4,
              f"eval[{d-1}]={evals[d-1]:.1e}, eval[{d}]={evals[d]:.2e}")
        check(f"d={d}: codewords dark under all four V_k", res < 1e-4,
              f"max residual {res:.1e}")
        check(f"d={d}: labels match physical combs", comb_ov > 0.9,
              f"min ansatz overlap {comb_ov:.3f}")
        print(f"        n_bar = {nbar:.3f}", flush=True)

    # ---------------- spectrum checks + bench at N_SPEC[d] ----------------
    print(flush=True)
    words, jumps, gaps = {}, {}, {}
    for d in (2, 3):
        Nd = N_SPEC[d]
        V = gkp_jumps(Nd, d, EPS)
        ws, evals = gkp_codewords(Nd, d, EPS, V)
        words[d], jumps[d] = ws, V
        c_stab = [np.sqrt(GAMMA) * v for v in V]
        Lsuper = qt.liouvillian(0 * c_stab[0], c_stab)
        rates = np.sort(-scipy.linalg.eigvals(Lsuper.full()).real)
        nz = int(np.sum(rates < 1e-6))
        gap = rates[nz] if nz < len(rates) else np.nan
        gaps[d] = gap
        pred = 2.0 * np.pi * d * EPS * GAMMA
        check(f"d={d}: d^2={d*d} zero Liouvillian modes (N={Nd})",
              nz == d * d, f"found {nz}")
        check(f"d={d}: confinement gap ~ eta^2 eps",
              0.1 * pred < gap < 10 * pred,
              f"gap {gap:.3f} vs pred {pred:.2f}")

    if not OK:
        print("\nCARD: FAIL — bench not run.", flush=True)
        return
    print("\nCARD: PASS\n", flush=True)

    print(f"worst logical (k1={K1}, kphi={KPHI}, GAMMA={GAMMA}):", flush=True)
    print(f"{'noise':<18s} {'d=2':>11s} {'d=3':>11s} {'penalty':>9s}",
          flush=True)
    results = {}
    for label, k1, kphi in (("loss only", K1, 0.0),
                            ("dephasing only", 0.0, KPHI),
                            ("mixed", K1, KPHI)):
        vals = {}
        for d in (2, 3):
            Nd = N_SPEC[d]
            c = [np.sqrt(GAMMA) * v for v in jumps[d]]
            if k1:
                c.append(np.sqrt(k1) * qt.destroy(Nd))
            if kphi:
                c.append(np.sqrt(kphi) * qt.num(Nd))
            vals[d] = dense_logical_fast(c, weyl_set(words[d]),
                                         ceiling=0.5 * gaps[d])
        pen = vals[3] / vals[2] if vals[2] > 0 else float("inf")
        results[label] = vals
        print(f"{label:<18s} {vals[2]:11.2e} {vals[3]:11.2e} {pen:8.2f}x",
              flush=True)

    # ---------------- time-domain cross-check (mixed, both d) --------------
    print("\ntime-domain cross-check (mixed noise, N_chk space):", flush=True)
    for d in (2, 3):
        ws = words_chk[d]
        V = gkp_jumps(N_CHK, d, EPS)
        c = [np.sqrt(GAMMA) * v for v in V] + \
            [np.sqrt(K1) * qt.destroy(N_CHK), np.sqrt(KPHI) * num_chk]
        X = sum(ws[(mu + 1) % d] * ws[mu].dag() for mu in range(d))
        psi = sum(ws) .unit()
        w_ref = results["mixed"][d]
        t_max = min(0.5 / max(w_ref, 1e-9), 5000.0)
        ts = np.linspace(0.0, t_max, 25)
        opt = {"nsteps": 200000, "method": "bdf"}
        res = qt.mesolve(0 * c[0], psi, ts, c, e_ops=[X], options=opt)
        y = np.abs(np.array(res.expect[0]))
        good = y > max(1e-4, 1e-3 * y[0])
        fit = (-np.polyfit(ts[good], np.log(y[good] + 1e-30), 1)[0]
               if good.sum() > 3 else float("nan"))
        print(f"  d={d}: X-coherence fit {fit:.2e} vs extractor "
              f"{w_ref:.2e} (ratio {fit / w_ref if w_ref else np.nan:.2f})",
              flush=True)

    print("\nReading: mixed-noise penalty <= ~3x -> GKP qudits pass the "
          "practicality bar (first pure-Lindblad qudit-GKP numerics either "
          "way). Penalty ~ pair-cat's 7-14x -> dimension cost is encoding-"
          "independent. Cross-check ratios ~1 required for trust.", flush=True)


if __name__ == "__main__":
    main()
