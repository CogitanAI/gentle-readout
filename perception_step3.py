"""The fair readability probe: matched n_bar + QND self/conjugate split.

Two refinements over perception_frontier.py:
1. MATCHED n_bar. Step 2 ran cats at n_bar~4 vs GKP~1.9 (cats had 2x photons =
   unfair readability edge). Here all codes sit at GKP's native n_bar~1.9
   (cats brought down to alpha=sqrt(1.9)~1.378); n_bar reported per code so the
   match is auditable. Cats gated on conditioning (orthonormal + dark); a code
   that fails at low alpha is skipped, not fudged.
2. CONJUGATE-SPLIT disturbance. Step 2's Disturb = worst logical rate (aggregate)
   couldn't tell "reads Z but corrupts Z" from "reads Z, only back-acts on X".
   Here, for observable M we extract the induced rate on BOTH logical axes:
     reading Z via M:  self = rate_Z  (BAD: corrupts what you read)
                       conj = rate_X  (OK : unavoidable back-action)
   A genuine QND / self-reporting readout has self ~ 0. The decisive metric:
     P_QND(read Z via M) = Perceive(M,Z) / rate_Z   (info per SELF-damage)
   The sharp claim, sharpened: GKP has a modular observable that reads a logical
   op with self ~ 0 (QND) where the cat's best readout self-disturbs.

Per-op rates via a lean sparse shift-invert extractor (no steadystate — GKP's
four-jump manifold is d^2-degenerate and would break qt.steadystate).

Run:  python perception_step3.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import qutip as qt
from scipy.sparse.linalg import eigs

from perception_frontier import (GAMMA, KAPPA1, cat_code, gkp_code,
                                  observables, perceive)

TARGET_NBAR = 1.9


def induced_rates(c_ops, logops, rate_ceiling=0.5):
    """Per-operator logical rate from stabilizer+gamma*D[M]; sparse shift-invert,
    no steadystate. Returns list aligned with logops."""
    dim = c_ops[0].shape[0]
    Lsuper = qt.liouvillian(qt.qzero(c_ops[0].dims[0]), c_ops)
    Ls = Lsuper.to("CSR").data_as("csr_matrix")
    k = min(30, dim * dim - 2)
    vals, vecs = eigs(Ls, k=k, sigma=-1e-9, which="LM",
                      return_eigenvectors=True, maxiter=max(400, 4 * Ls.shape[0]))
    rates = [-vals[i].real for i in range(len(vals))]
    mats = [vecs[:, i].reshape(dim, dim, order="F") for i in range(len(vals))]
    norms = [np.linalg.norm(m) + 1e-15 for m in mats]
    out = []
    for O in logops:
        Od = O.full()
        best_rate, best_ov = 0.0, -1.0
        for i in range(len(vals)):
            if rates[i] < 1e-9 or rates[i] > rate_ceiling:
                continue
            ov = abs(np.trace(Od.conj().T @ mats[i])) / norms[i]
            if ov > best_ov:
                best_ov, best_rate = ov, rates[i]
        out.append(best_rate if best_ov > 0.1 else 0.0)
    return out


def nbar_of(code):
    n = qt.num(code["N"])
    return float(np.mean([qt.expect(n, s) for s in code["z_states"]]))


def conditioned(code):
    zs = code["z_states"]
    d = len(zs)
    ov = max((abs(zs[i].overlap(zs[j])) for i in range(d)
              for j in range(i + 1, d)), default=0.0)
    stab = code["stab"]
    res = max((sum(c * z for c in stab)).norm() / max(len(stab), 1) for z in zs)
    return ov, res


def scan(code):
    N, d = code["N"], code["d"]
    logops = code["logops"]
    XL = logops[0]
    if code["name"].startswith("GKP"):
        ZL = logops[1]              # GKP: [X, Z, XZ]
    elif len(logops) == 3:
        ZL = logops[2]              # cat d=2: [X, Y, Z]
    else:
        ZL = logops[1]              # cat d>2: [X, Z]
    z_st, x_st = code["z_states"], code["x_states"]
    obs, _ = observables(N, d)
    nb = nbar_of(code)
    ov, res = conditioned(code)
    print(f"\n=== {code['name']}  (d={d}, N={N})  n_bar={nb:.2f}  "
          f"[cond ov={ov:.1e} res={res:.1e}] ===", flush=True)
    if ov > 0.12 or res > 0.1:
        print("  ILL-CONDITIONED at this n_bar — skipping (unfair to score).",
              flush=True)
        return None
    print(f"{'obs':>8s} {'Perc_Z':>9s} {'Perc_X':>9s} {'self_Z':>9s} "
          f"{'conj_X':>9s} {'P_QND_Z':>9s}   note", flush=True)
    rows = []
    for nm, M in obs.items():
        pz, px = perceive(M, z_st), perceive(M, x_st)
        rZ, rX = induced_rates(code["stab"] + [np.sqrt(GAMMA) * M], [ZL, XL])
        pqnd = pz * GAMMA / rZ if rZ > 1e-9 else (np.inf if pz > 1e-6 else 0.0)
        note = ""
        if pz > 1e-3 and rZ < 1e-6:
            note = "QND read-Z"
        rows.append((nm, pz, px, rZ, rX, pqnd))
        pq = f"{pqnd:9.2e}" if np.isfinite(pqnd) else "     QND!"
        print(f"{nm:>8s} {pz:9.2e} {px:9.2e} {rZ:9.2e} {rX:9.2e} {pq}   {note}",
              flush=True)
    # environment reference (self/conj under loss and dephasing)
    for envnm, M in (("loss", code["a"]), ("deph", qt.num(N))):
        rZ, rX = induced_rates(code["stab"] + [np.sqrt(KAPPA1) * M], [ZL, XL])
        print(f"{'env:'+envnm:>8s} {'-':>9s} {'-':>9s} {rZ:9.2e} {rX:9.2e}",
              flush=True)
    best = max((r for r in rows if np.isfinite(r[5])),
               key=lambda r: r[5], default=None)
    qnd = [r for r in rows if r[1] > 1e-3 and r[3] < 1e-6]
    if qnd:
        print(f"  --> QND read-Z observable(s): {[r[0] for r in qnd]}  "
              f"(read Z with self-disturbance ~0)", flush=True)
    elif best:
        print(f"  --> best read-Z: {best[0]} P_QND={best[5]:.2e} "
              f"(self_Z={best[3]:.2e} > 0 -> not QND)", flush=True)
    return rows


def main():
    print(f"FAIR PERCEPTION PROBE  target n_bar={TARGET_NBAR}  gamma={GAMMA}",
          flush=True)
    alpha = np.sqrt(TARGET_NBAR)
    for d in (2, 3, 4):
        scan(cat_code(alpha, d))
    for d, Ngkp in ((2, 80), (3, 90)):
        g = gkp_code(d, Ngkp)
        if not g.get("kernel_ok", True):
            print(f"\n=== GKP d={d}: kernel dirty at N={Ngkp}, skip ===",
                  flush=True)
            continue
        scan(g)
    print("\nReading: a 'QND read-Z' observable (Perc_Z>0, self_Z~0) = readout "
          "that sees the logical WITHOUT corrupting it — the real 'readable-"
          "while-protected'. Claim confirmed if GKP has one via mod-q/mod-p and "
          "the matched-n_bar cats do NOT.", flush=True)


if __name__ == "__main__":
    main()
