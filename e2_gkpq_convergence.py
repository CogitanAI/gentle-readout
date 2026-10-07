"""Is the GKP-q cell converged in Fock cutoff and time window?

(Numbering in this docstring follows the 2026-08-21 manuscript; in the
2026-10-06 revision the verification grid is Table III and the flux table
this produces is Table II, Sec. III.A.)


The manuscript's largest looseness figure, "316x", is GKP-q: Gamma_self = 3.1e-3
against a bound of 9.8e-6. Two things are known to be soft about that cell:
  * the windowed slope had not stopped falling at the last window (t = 4):
    2.69e-2 -> 3.26e-3, and stable() returned the last value;
  * q is unbounded, so the result could depend on the Fock cutoff N, and the
    bound terms (kappa^2, Lambda_01) could move with N as well as the rate.

For each cutoff N this reports:
  1. kernel_ok and the bound terms 2 gamma kappa^2, gamma eps_KL^2 / 2;
  2. the EXACT logarithmic slope Gamma(t) = -Re tr[Z^dag L rho(t)] / Re tr[Z^dag rho(t)]
     with rho(0) = Z P / 2, on a log grid to t = 512 (1/Gamma ~ 330), from one dense
     eigendecomposition -- no window, no finite differences;
  3. the dense top-overlap spectral rate (the gamma_sweep.py estimator) with the
     overlap-tie ratio, for N where dense fits in memory;
  4. the truncation-edge weight: population in the top 5 Fock levels at t = 64,
     starting from logical |0>. A meter that pushes weight to the edge fakes
     convergence.

PASS (from the plan): Gamma_self stable to 5% across cutoffs. If it moves by more
than 20%, the 316x changes everywhere it is quoted.

Run:  python -u e2_gkpq_convergence.py [N ...]   (writes e2_gkpq_convergence.json)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import qutip as qt
import scipy.linalg
import scipy.sparse as sp

from perception_frontier import gkp_code, observables

GAMMA = 0.05
TS = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
DENSE_MAX_N = 110
CEIL = 0.5
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2_gkpq_convergence.json")


def lam_split(M, ws, Pcode):
    """kappa^2, |Lambda_01|^2, tr Lambda -- copied from release_v1.2/gamma_sweep.py."""
    N = M.shape[0]
    Pp = qt.qeye(N) - Pcode
    L = np.array([[complex(ws[i].overlap(M * (Pp * (M * ws[j]))))
                   for j in range(2)] for i in range(2)])
    return (abs(complex(ws[0].overlap(M * ws[1]))) ** 2,
            abs(L[0, 1]) ** 2, float(np.real(np.trace(L))))


def vec(op):
    return op.full().reshape(-1, order="F")


def spectral_all(Ls, Z, w0, N, floor=1e-12, top=5, t_edge=64.0):
    """One dense eigendecomposition gives everything:
       * the exact log-slope -d/dt ln C(t) at every t in TS, with rho(0) = Z P/2
         (same estimator as autocorr_rates_fixed, extended to long times);
       * the top-overlap spectral rate and its tie ratio (gamma_sweep.py);
       * the truncation-edge weight of rho(t_edge) from logical |0>.
    The sparse time-stepping alternative is hopeless here: the GKP stabilizer
    makes ||L||_1 ~ 7e6, so expm_multiply needs millions of steps."""
    vals, vecs = scipy.linalg.eig(Ls.toarray())
    rates = -vals.real
    lz = vec(Z).conj()
    rhs = np.stack([vec(Z * Pcode_g / 2), vec(w0 * w0.dag())], axis=1)
    coef = scipy.linalg.solve(vecs, rhs)
    lam = np.where(vals.real > 0, 1j * vals.imag, vals)
    w = (lz @ vecs) * coef[:, 0]
    slopes = []
    for t in TS:
        e = np.exp(lam * t)
        C = complex(np.sum(w * e))
        dC = complex(np.sum(w * lam * e))
        slopes.append(dict(t=t, C=C.real, rate=-(dC.real) / C.real))
    rho = (vecs @ (coef[:, 1] * np.exp(lam * t_edge))).reshape(N, N, order="F")
    pop = np.real(np.diag(rho))
    edge = float(pop[-top:].sum() / pop.sum())
    ovs = np.abs(lz @ vecs) / (np.linalg.norm(vecs, axis=0) + 1e-15)
    mask = (rates > floor) & (rates < CEIL)
    idx = np.argsort(np.where(mask, ovs, -1.0))[::-1][:3]
    top3 = [(float(ovs[i]), float(rates[i])) for i in idx]
    return slopes, top3, edge


def run(N):
    t0 = time.time()
    code = gkp_code(2, N)
    ws, Pcode, Z = code["z_states"], code["Pcode"], code["logops"][1]
    M = observables(N, 2)[0]["q"]
    k2, off2, trL = lam_split(M, ws, Pcode)
    b1, b2 = 2 * GAMMA * k2, GAMMA * off2 / (2 * trL)
    c_ops = code["stab"] + [np.sqrt(GAMMA) * M]
    Ls = sp.csr_matrix(qt.liouvillian(0 * M, c_ops).to("CSR").data.as_scipy())
    rec = dict(N=N, kernel_ok=bool(code["kernel_ok"]), b_kappa=b1, b_kl=b2,
               bound=max(b1, b2), trLambda=trL)
    global Pcode_g
    Pcode_g = Pcode
    td = time.time()
    slopes, top, edge = spectral_all(Ls, Z, ws[0], N)
    rec.update(slopes=slopes, edge_weight_t64=edge, dense_top3=top,
               dense_rate=top[0][1],
               tie=top[1][0] / top[0][0] if top[0][0] > 0 else float("nan"))
    print(f"N={N} kernel_ok={rec['kernel_ok']}  2gk^2={b1:.3e}  geps^2/2={b2:.3e}  "
          f"edge(t=64)={edge:.2e}  [{(time.time()-td)/60:.1f} min]", flush=True)
    print("   t:    " + "  ".join(f"{x['t']:>8g}" for x in slopes), flush=True)
    print("   rate: " + "  ".join(f"{x['rate']:8.2e}" for x in slopes), flush=True)
    print("   C(t): " + "  ".join(f"{x['C']:8.2e}" for x in slopes), flush=True)
    print(f"   dense: rate={top[0][1]:.4e} (ovl {top[0][0]:.3f})  "
          f"2nd: {top[1][1]:.3e} (ovl {top[1][0]:.3f})  tie={rec['tie']:.2f}", flush=True)
    rec["secs"] = time.time() - t0
    return rec


def main():
    Ns = [int(a) for a in sys.argv[1:]] or [80, 90, 100, 110]
    out = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["N"] for r in out}
    print(f"E2: GKP-q convergence, gamma={GAMMA}, N={Ns}", flush=True)
    for N in Ns:
        if N in done:
            continue
        out.append(run(N))
        out.sort(key=lambda r: r["N"])
        json.dump(out, open(OUT, "w"), indent=1)
    print("\nN    bound       late-time rate (t=512)  dense     R_dense", flush=True)
    for r in out:
        late = r["slopes"][-1]["rate"]
        dr = r.get("dense_rate", float("nan"))
        print(f"{r['N']:<4d} {r['bound']:.3e}   {late:.3e}               {dr:.3e}  "
              f"{dr / r['bound']:.0f}x", flush=True)


if __name__ == "__main__":
    main()
