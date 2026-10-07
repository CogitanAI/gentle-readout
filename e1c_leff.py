"""Is the residual of e1_flux_decomposition.py the initial-slope vs slow-mode gap
of the exact leading-order generator?

(Numbering in this docstring follows the 2026-08-21 manuscript; in the
2026-10-06 revision the verification grid is Table III and the flux table
this produces is Table II, Sec. III.A.)


L_eff(rho) = gamma [ D[M_c] rho - 1/2 {Lambda, rho} + R(E rho E^dag) ]   (Eq. 11)
on 2x2 code-space operators, with R the cascaded return channel of E1.
Eq. (12) is its INITIAL slope on Z. The spectral sweep measures its SLOW MODE.
Compare L_eff's top-Z-overlap eigenvalue (per unit gamma) against the spectral
rate at gamma = 1.953125e-4 (all cells inside (G3)), per unit gamma.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from perception_frontier import cat_code, gkp_code, observables
from e1_flux_decomposition import split_basis, return_channel

HERE = os.path.dirname(os.path.abspath(__file__))
sw = json.load(open(os.path.join(HERE, "data", "gamma_sweep_results.json")))
G = 1.953125e-4
spec = {( {"cat d=2": "cat", "GKP d=2": "GKP"}[r["code"]], r["meter"]): r["gamma_self"] / G
        for r in sw if isinstance(r, dict) and r.get("gamma") == G}
e1 = {(r["code"], r["meter"]): r for r in json.load(open(os.path.join(HERE, "e1_flux_decomposition.json")))}

def D(L, r):
    return L @ r @ L.conj().T - 0.5 * (L.conj().T @ L @ r + r @ L.conj().T @ L)

print(f"{'cell':<12}{'init slope':>11}{'L_eff slow':>11}{'spectral':>10}{'slow/spec':>10}{'init/spec':>10}{'ovl2/ovl1':>10}")
out = []
for name, code in (("cat", cat_code(np.sqrt(1.9), 2)), ("GKP", gkp_code(2, 80))):
    N = code["N"]
    Vc, Vp = split_basis(code["z_states"], N)
    R, _, _ = return_channel(code["stab"], Vc, Vp)
    obs, _ = observables(N, 2)
    Z = np.diag([1.0, -1.0]).astype(complex)
    for meter in ["parity", "mod-q", "mod-p", "q", "p", "n"]:
        if (name, meter) == ("GKP", "parity"):
            continue
        Mm = obs[meter].full()
        Mc = Vc.conj().T @ Mm @ Vc
        E = Vp.conj().T @ Mm @ Vc
        Lam = E.conj().T @ E
        cols = []
        for k in range(4):
            r = np.zeros(4, complex); r[k] = 1; r = r.reshape(2, 2, order="F")
            Lr = D(Mc, r) - 0.5 * (Lam @ r + r @ Lam) + R(E @ r @ E.conj().T)
            cols.append(Lr.reshape(-1, order="F"))
        Leff = np.column_stack(cols)                  # per unit gamma
        vals, vecs = np.linalg.eig(Leff)
        lz = Z.reshape(-1, order="F").conj()
        ov = np.abs(lz @ vecs) / np.linalg.norm(vecs, axis=0)
        mask = -vals.real > 1e-12
        idx = np.argsort(np.where(mask, ov, -1))[::-1]
        slow = -vals[idx[0]].real
        init = -np.real(lz @ (Leff @ (Z / 2).reshape(-1, order="F")))   # Tr[Z rho0]=1
        sp = spec[(name, meter)]
        tie = ov[idx[1]] / ov[idx[0]]
        out.append(dict(code=name, meter=meter, init=init, slow=slow, spectral=sp, tie=float(tie),
                        eigs=[complex(v) for v in vals]))
        print(f"{name+' '+meter:<12}{init:11.4e}{slow:11.4e}{sp:10.4e}{slow/sp:10.4f}{init/sp:10.4f}{tie:10.2f}", flush=True)
json.dump([{**o, "eigs": [[v.real, v.imag] for v in o["eigs"]]} for o in out],
          open(os.path.join(HERE, "e1c_leff.json"), "w"), indent=1)
