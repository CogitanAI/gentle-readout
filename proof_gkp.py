"""Machine verification of every step of the GKP tuned-zero lemma.

Construction (comb-envelope finite-energy GKP, as in nogo_sweep.gkp):
  chi_mu(q) = sum_s f((2s+mu) l) g_v(q - (2s+mu) l),  l = sqrt(pi),
  g_v = Gaussian wavefunction with Var q = v = Delta^2/2,
  f(x) = exp(-Delta^2 x^2 / 2) (even, positive).
Codewords |0'>,|1'> = Lowdin (symmetric) orthonormalization of chi_0, chi_1.
Meter M_phi = cos(lam q + phi), lam = pi/l = sqrt(pi).

Identities to verify (E* exact for the construction; residuals should be at
the Fock-truncation floor ~1e-13):

 E1  <chi0|sin(lam q)|chi1> = 0, <chi0|sin(2 lam q)|chi1> = 0   (parity)
 E2  <chi0|cos(lam q)|chi1> = 0            (midpoint phases cos((2k+1)pi/2)=0)
 E3  D_mu := <mu|cos(2 lam q)|mu>_norm = e^{-2 lam^2 v}   (all midpoint
     phases +1; norm-sum collapses)
 E4  <0|cos(2 lam q)|1>_norm = -delta e^{-2 lam^2 v}      (all midpoint
     phases -1; same sum as the overlap delta)
 E5  kappa(phi) = A cos(phi) with A = cd (S0 + S1),
     S_mu = <mu|cos(lam q)|mu>_norm; c,d Lowdin coefficients
 E6  C := -1/2 <0'|cos(2 lam q)|1'> = delta e^{-2 lam^2 v} / (1 - delta^2)
 E7  Lambda01(phi) = -C cos(2 phi) - [A^2 (c^2+d^2)/(cd)] cos^2(phi)
 E8  Dm(phi) = cos(phi) (S0 - S1) / sqrt(1 - delta^2)
 A1  (asymptotics, not exact) delta ~ 2 e^{-pi/(4 Delta^2)},
     S0 - S1 ~ 2 e^{-pi Delta^2 / 4},  Dm(3pi/4) ~ -sqrt2 e^{-pi Delta^2/4}
"""
import json
import numpy as np
import nogo_sweep as ns

def signed_data(cell, params):
    f = np.zeros_like(cell.x)
    for (amp, lam, phi) in params:
        f = f + amp * np.cos(lam * cell.x + phi)
    f0, f1 = f * cell.u0, f * cell.u1
    M00 = np.vdot(cell.u0, f0).real
    M11 = np.vdot(cell.u1, f1).real
    M01 = np.vdot(cell.u0, f1)
    W01 = np.vdot(f0, f1)
    L01 = W01 - (M00 * M01 + M01 * M11)
    Dm = M00 - M11
    return M01, L01, Dm

def raw_combs(N, Delta):
    """Pre-Lowdin normalized comb states (same teeth as ns.gkp)."""
    lq = np.sqrt(np.pi)
    r = -np.log(Delta)
    sq = ns.squeezed_vac(N, r)
    out = []
    for mu in (0, 1):
        v = np.zeros(N, dtype=complex)
        smax = int(np.ceil(6.0 / (Delta * lq))) + 2
        for s in range(-smax, smax + 1):
            x = (2 * s + mu) * lq
            w = np.exp(-0.5 * Delta ** 2 * x ** 2)
            if w < 1e-14:
                continue
            v = v + w * ns.displace_q(N, x, sq)
        out.append(v / np.linalg.norm(v))
    return out

def op_in_qbasis(md, func):
    return md.Uq @ np.diag(func(md.xq)) @ md.Uq.conj().T

RES = {}
lam = np.sqrt(np.pi)
for N in (140, 200):
    md = ns.mode(N)
    cosl = op_in_qbasis(md, lambda x: np.cos(lam * x))
    sinl = op_in_qbasis(md, lambda x: np.sin(lam * x))
    cos2l = op_in_qbasis(md, lambda x: np.cos(2 * lam * x))
    sin2l = op_in_qbasis(md, lambda x: np.sin(2 * lam * x))
    for Delta in (0.30, 0.35, 0.40, 0.50):
        v = Delta ** 2 / 2
        e2 = np.exp(-2 * lam ** 2 * v)          # e^{-2 lam^2 v} = e^{-pi D^2}
        chi0, chi1 = raw_combs(N, Delta)
        delta = np.vdot(chi0, chi1).real
        # Lowdin coefficients
        u_, w_ = (1 + delta) ** -0.5, (1 - delta) ** -0.5
        c_, d_ = (u_ + w_) / 2, (u_ - w_) / 2
        S0 = np.vdot(chi0, cosl @ chi0).real
        S1 = np.vdot(chi1, cosl @ chi1).real
        D0 = np.vdot(chi0, cos2l @ chi0).real
        D1 = np.vdot(chi1, cos2l @ chi1).real
        X01 = np.vdot(chi0, cos2l @ chi1).real
        # exact-identity residuals
        r = {}
        r['E1_sin'] = abs(np.vdot(chi0, sinl @ chi1))
        r['E1_sin2'] = abs(np.vdot(chi0, sin2l @ chi1))
        r['E2_cos_raw'] = abs(np.vdot(chi0, cosl @ chi1))
        r['E3_D0'] = abs(D0 - e2)
        r['E3_D1'] = abs(D1 - e2)
        r['E4_cross'] = abs(X01 - (-delta * e2))
        # Lowdin codewords (must match ns.gkp output up to global sign)
        l0 = c_ * chi0 + d_ * chi1
        l1 = d_ * chi0 + c_ * chi1
        g0, g1 = ns.gkp(N, Delta)
        r['lowdin_match'] = min(np.linalg.norm(l0 - g0), np.linalg.norm(l0 + g0))
        cell = ns.ModCell(md, l0, l1, 'q')
        A = c_ * d_ * (S0 + S1)
        C = delta * e2 / (1 - delta ** 2)
        Cmeas = -0.5 * np.vdot(l0, cos2l @ l1).real
        r['E6_C'] = abs(Cmeas - C)
        # E5, E7, E8 across a phi grid
        e5, e7, e8 = [], [], []
        B2 = A ** 2 * (c_ ** 2 + d_ ** 2) / (c_ * d_)
        for phi in np.linspace(0, np.pi, 13):
            M01, L01, Dm = signed_data(cell, [(1.0, lam, phi)])
            e5.append(abs(M01.real - A * np.cos(phi)))
            e7.append(abs(L01.real - (-C * np.cos(2 * phi)
                                      - B2 * np.cos(phi) ** 2)))
            e8.append(abs(Dm - np.cos(phi) * (S0 - S1)
                          / np.sqrt(1 - delta ** 2)))
        r['E5_kappa_law'] = max(e5)
        r['E7_L01_law'] = max(e7)
        r['E8_Dm_law'] = max(e8)
        # values + asymptotics
        r['val_delta'] = delta
        r['val_A'] = A
        r['val_C'] = C
        r['val_S0mS1'] = S0 - S1
        r['A1_delta_asym'] = delta / (2 * np.exp(-np.pi / (4 * Delta ** 2)))
        r['A1_S_asym'] = (S0 - S1) / (2 * np.exp(-np.pi * Delta ** 2 / 4))
        Dm34 = np.cos(3 * np.pi / 4) * (S0 - S1) / np.sqrt(1 - delta ** 2)
        r['val_Dm_3pi4'] = Dm34
        r['A1_Dm_asym'] = Dm34 / (-np.sqrt(2) * np.exp(-np.pi * Delta ** 2 / 4))
        RES[f'N{N}_D{Delta}'] = {k: float(np.real(x)) for k, x in r.items()}
        print(f"N={N} D={Delta}: "
              f"E1={r['E1_sin']:.1e}/{r['E1_sin2']:.1e} "
              f"E2={r['E2_cos_raw']:.1e} E3={max(r['E3_D0'],r['E3_D1']):.1e} "
              f"E4={r['E4_cross']:.1e} E5={r['E5_kappa_law']:.1e} "
              f"E6={r['E6_C']:.1e} E7={r['E7_L01_law']:.1e} "
              f"E8={r['E8_Dm_law']:.1e}", flush=True)
        print(f"   delta={delta:.6e} (asym ratio {r['A1_delta_asym']:.4f})  "
              f"C={C:.6e}  A={A:.3e}  "
              f"Dm(3pi/4)={Dm34:.6f} (asym ratio {r['A1_Dm_asym']:.4f})",
              flush=True)

json.dump(RES, open('proof_gkp_results.json', 'w'), indent=1)
print("-> proof_gkp_results.json", flush=True)
