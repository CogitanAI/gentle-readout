"""Does the flux identity explain the looseness of Theorem 1?

(Numbering in this docstring follows the 2026-08-21 manuscript; in the
2026-10-06 revision the verification grid is Table III and the flux table
this produces is Table II, Sec. III.A.)


Theorem 1's leak term is a converse: each leak event costs at least the Helstrom
floor, so Gamma_self >= gamma |Lambda_01|^2 / (2 tr Lambda). Table II shows the
bound loose by 4x to 316x wherever that term binds alone. The manuscript says what
sets the measured rate there "is not established here".

Appendix C, Eq. (12), is not a bound. It is the exact leading-order accounting
    Gamma_self ~= 2 gamma kappa^2 + gamma (eps_0^2 delta_0 + eps_1^2 delta_1),
    delta_mu = 1 - <mu| R(|w_mu><w_mu|) |mu>,   |w_mu> = E|mu> / eps_mu,
with R the stabilizer's ACTUAL return channel, Eq. (16):
    R(sigma) = int_0^inf J_c( e^{L'_perp tau} sigma ) dtau = J_c( -L'_perp^{-1} sigma ),
    L'_perp = -1/2 {K_perp, .} + J_perp,   J_perp(s) = sum A_k s A_k^dag,
    J_c(s) = sum B_k s B_k^dag,   A_k = P_perp L_k P_perp,   B_k = P L_k P_perp.
This is the cascaded return channel: both the cat stabilizer (a^2 - alpha^2) and
the GKP stabilizer move leaked population through other leaked states, so the
"one return application" shortcut in code_release/lemma_A_validation.py (valid for
direct return only) does not apply.

If Eq. (12) reproduces the measured Gamma_self, the looseness is fully accounted
for, and it splits into two named factors per cell:
    recovery inefficiency  = (eps_0^2 d_0 + eps_1^2 d_1) / (eps_0^2 dH_0 + eps_1^2 dH_1)
                             (actual stabilizer vs the Helstrom-optimal floor)
    bounding-chain loss    = (eps_0^2 dH_0 + eps_1^2 dH_1) / (|Lambda_01|^2 / (2 tr Lambda))
where dH is each codeword's share of the Helstrom floor (1 - sqrt(1-|s|^2))/2.

PASS (from the plan): within 15% at every leak-dominated cell, within 5% at
cat-parity and GKP-mod-p. KILL: any cell off by more than 2x means a damage channel
that Eq. (12) does not contain.

Run:  python -u e1_flux_decomposition.py [cat|gkp|all]
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

GAMMA = 0.05
METERS = ["parity", "mod-q", "mod-p", "q", "p", "n"]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e1_flux_decomposition.json")

# Table II as printed in the submitted manuscript (measured Gamma_self)
TABLE2 = {("cat", "parity"): 1.0e-1, ("cat", "mod-q"): 1.7e-3, ("cat", "mod-p"): 9.4e-3,
          ("cat", "q"): 1.2e-3, ("cat", "p"): 1.5e-3, ("cat", "n"): 4.8e-3,
          ("GKP", "parity"): 1.4e-8, ("GKP", "mod-q"): 1.7e-3, ("GKP", "mod-p"): 7.4e-2,
          ("GKP", "q"): 3.1e-3, ("GKP", "p"): 5.0e-2, ("GKP", "n"): 1.1e-1}


def split_basis(ws, N):
    """Orthonormal code basis Vc (the Z-eigenstate codewords) and its complement Vp."""
    Vc = np.column_stack([w.full().ravel() for w in ws])   # column mu is |mu>
    Pc = Vc @ Vc.conj().T
    u, s, _ = np.linalg.svd(np.eye(N) - Pc)
    Vp = u[:, : N - Vc.shape[1]]
    return Vc, Vp


def return_channel(stab, Vc, Vp):
    """Return R as a function on perp-block operators (in the Vp basis)."""
    Ls = [L.full() for L in stab]
    A = [Vp.conj().T @ L @ Vp for L in Ls]
    B = [Vc.conj().T @ L @ Vp for L in Ls]
    n = Vp.shape[1]
    I = np.eye(n)
    Kp = Vp.conj().T @ sum(L.conj().T @ L for L in Ls) @ Vp
    # column-stacking: vec(A X B) = (B^T kron A) vec(X)
    Lp = -0.5 * (np.kron(I, Kp) + np.kron(Kp.T, I))
    for a in A:
        Lp = Lp + np.kron(a.conj(), a)
    lu = scipy.linalg.lu_factor(Lp)
    # G2 diagnostic: smallest eigenvalue of the drain operator G = sum B^dag B
    G = sum(b.conj().T @ b for b in B)
    g_drain = float(np.linalg.eigvalsh((G + G.conj().T) / 2).min())
    g_K = float(np.linalg.eigvalsh((Kp + Kp.conj().T) / 2).min())

    def R(sig):
        X = scipy.linalg.lu_solve(lu, -sig.reshape(-1, order="F")).reshape(n, n, order="F")
        return sum(b @ X @ b.conj().T for b in B)
    return R, g_drain, g_K


def cell(code_name, code, zli, meter, M, R, Vc, Vp):
    N = M.shape[0]
    Mm = M.full()
    Mc = Vc.conj().T @ Mm @ Vc
    kappa2 = abs(Mc[0, 1]) ** 2
    Ewav = Vp.conj().T @ Mm @ Vc               # E = P_perp M P, columns E|mu> in Vp basis
    eps = np.linalg.norm(Ewav, axis=0)
    Lam = Ewav.conj().T @ Ewav
    trL = float(np.real(np.trace(Lam)))
    off2 = abs(Lam[0, 1]) ** 2
    b_kappa, b_kl = 2 * GAMMA * kappa2, (GAMMA * off2 / (2 * trL) if trL > 1e-14 else 0.0)
    out = dict(code=code_name, meter=meter, kappa2=kappa2, eps0=float(eps[0]),
               eps1=float(eps[1]), Lambda01_abs=float(np.sqrt(off2)), trLambda=trL,
               b_kappa=b_kappa, b_kl=b_kl, bound=max(b_kappa, b_kl))
    deltas, trR, dH = [], [], []
    s = (Lam[0, 1] / (eps[0] * eps[1])) if eps.min() > 1e-12 else 0.0
    for mu in range(2):
        if eps[mu] < 1e-12:
            deltas.append(0.0); trR.append(float("nan")); dH.append(0.0)
            continue
        w = Ewav[:, mu] / eps[mu]
        rec = R(np.outer(w, w.conj()))
        trR.append(float(np.real(np.trace(rec))))
        deltas.append(float(1 - np.real(rec[mu, mu]) / np.real(np.trace(rec))))
        dH.append((1 - np.sqrt(max(0.0, 1 - abs(s) ** 2))) / 2)
    flux = GAMMA * (eps[0] ** 2 * deltas[0] + eps[1] ** 2 * deltas[1])
    helst = GAMMA * (eps[0] ** 2 * dH[0] + eps[1] ** 2 * dH[1])
    pred = 2 * GAMMA * kappa2 + flux
    meas = TABLE2[(code_name, meter)]
    out.update(s_abs=float(abs(s)), delta=deltas, trR=trR, flux=float(flux),
               helstrom_flux=float(helst), predicted=float(pred), measured=meas,
               pred_over_meas=float(pred / meas),
               meas_over_bound=float(meas / out["bound"]) if out["bound"] > 1e-30 else float("nan"),
               recovery_inefficiency=float(flux / helst) if helst > 1e-30 else float("nan"),
               chain_loss=float(helst / b_kl) if b_kl > 1e-30 else float("nan"))
    return out


def run(which):
    codes = []
    if which in ("cat", "all"):
        codes.append(("cat", cat_code(np.sqrt(1.9), 2), 2))
    if which in ("gkp", "all"):
        codes.append(("GKP", gkp_code(2, 80), 1))
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    res = [r for r in res if r["code"] not in {c[0] for c in codes}]
    for name, code, zli in codes:
        t0 = time.time()
        N = code["N"]
        Vc, Vp = split_basis(code["z_states"], N)
        ortho = float(np.abs(Vc.conj().T @ Vc - np.eye(2)).max())
        dark = max(float(np.linalg.norm(L.full() @ Vc)) for L in code["stab"])
        R, g_drain, g_K = return_channel(code["stab"], Vc, Vp)
        print(f"\n{name} d=2  N={N}  codeword orthonormality err={ortho:.1e}  "
              f"dark-space residual max||L_k|mu>||={dark:.1e}", flush=True)
        print(f"  (G2) drain gap min eig G = {g_drain:.3e}   vs  min eig K_perp = {g_K:.3e}"
              f"   [{(time.time()-t0):.0f}s to build R]", flush=True)
        print(f"  {'meter':>7s} {'measured':>9s} {'predicted':>9s} {'pred/meas':>9s} "
              f"{'2gk^2':>9s} {'flux':>9s} {'meas/bnd':>8s} {'recov.ineff':>11s} "
              f"{'chain':>6s} {'trR':>11s}", flush=True)
        obs, _ = observables(N, 2)
        for meter in METERS:
            r = cell(name, code, zli, meter, obs[meter], R, Vc, Vp)
            r.update(N=N, g_drain=g_drain, g_K=g_K, dark_residual=dark)
            res.append(r)
            print(f"  {meter:>7s} {r['measured']:9.2e} {r['predicted']:9.2e} "
                  f"{r['pred_over_meas']:8.2f}x {r['b_kappa']:9.2e} {r['flux']:9.2e} "
                  f"{r['meas_over_bound']:7.0f}x {r['recovery_inefficiency']:10.1f}x "
                  f"{r['chain_loss']:5.1f}x {r['trR'][0]:5.3f}/{r['trR'][1]:5.3f}",
                  flush=True)
            json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
