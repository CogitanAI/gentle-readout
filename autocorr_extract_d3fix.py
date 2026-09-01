"""Fix for the d>2 autocorrelation NaN in autocorr_extract.py (audit follow-up).

ROOT CAUSE (diagnosed 2026-07-19): autocorr_rates builds the left vector as
lvec = O.dag().full().reshape(-1,'F').conj(), which contracts as tr[O rho],
so C(0) = tr[O O P]/d. For d=2 the logical Z is Hermitian and C(0)=1; for
d>2 the logical Z is the NON-HERMITIAN Weyl clock Z = sum_k omega^k |k><k|,
and tr[Z Z P]/d = (1/d) sum_k omega^{2k} = 0 exactly (roots of unity;
d=4: sum (-1)^k = 0). The abs(C0)>1e-12 guard then returns NaN at every t*.
Verified numerically: C0_original = 0j for cat d=3, cat d=4, GKP d=3, while
||Z - Z.dag|| = 3.5-4.3 (the operators really are non-Hermitian).

FIX: normalize with the adjoint -- C(t) = tr[Z^dag e^{Lt}(Z P/d)] / C0 with
C0 = tr[Z^dag Z P]/d (~1; slightly >1 for cats because the biorthogonal-dual
Weyl ops are only approximately unitary on the orthonormalized code basis).
For Hermitian O this is IDENTICAL to the original definition, so all d=2
paper numbers are untouched.

Also provides the per-pair Hermitian cross-check O_ij = Pi_i - Pi_j
(consistent with the paper's per-pair theorem scoping for d>2); reports the
worst pair rate.

New file only -- existing sources untouched. Usage:
  python -u autocorr_extract_d3fix.py diag      # reproduce + localize NaN
  python -u autocorr_extract_d3fix.py cat3      # cat d=3 via p   (paper 6.4e-3)
  python -u autocorr_extract_d3fix.py cat4      # cat d=4 via q   (paper 1.3e-2)
  python -u autocorr_extract_d3fix.py cat4lo    # cat d=4 at reduced cutoff (convergence)
  python -u autocorr_extract_d3fix.py table2    # all 12 Table II cells
  python -u autocorr_extract_d3fix.py gkp3 [N]  # GKP d=3 via mod-q (paper 2.1e-3)
"""
import sys
import numpy as np
import qutip as qt
import scipy.linalg

from perception_frontier import cat_code, gkp_code, observables, perceive

GAMMA = 0.05
TSTARS = (0.5, 1.0, 2.0, 4.0)


def autocorr_rates_fixed(c_ops, O, Pcode, d):
    """Windowed -dC/dt for (possibly non-Hermitian) operator O, with the
    ADJOINT-normalized autocorrelation C(t) = tr[O^dag e^{Lt}(O P/d)]/C0.
    Identical to autocorr_extract.autocorr_rates for Hermitian O."""
    Lsuper = qt.liouvillian(0 * c_ops[0], c_ops)
    rho0 = (O * Pcode / d).full().reshape(-1, order="F")
    # FIX: conj(vec(O)) @ vec(rho) = tr[O^dag rho]   (original used vec(O.dag()))
    lvec = O.full().reshape(-1, order="F").conj()
    vals, vecs = scipy.linalg.eig(Lsuper.full())
    vals = np.where(vals.real > 0, 1j * vals.imag, vals)
    c, *_ = np.linalg.lstsq(vecs, rho0, rcond=None)
    w = (lvec @ vecs) * c
    C0 = complex(np.sum(w))
    out = []
    for t in TSTARS:
        e = np.exp(vals * t)
        slope = -complex(np.sum(w * vals * e))
        # BUG FIX 2026-08-20: normalize by C(t), not C(0). Gamma_self is the
        # LOGARITHMIC derivative -d/dt ln C(t) = -Cdot(t)/C(t); dividing by C0
        # returns the raw derivative -Cdot(t), which is low by exp(-Gamma t).
        # Negligible where Gamma*t << 1, but 9.5% at cat/parity (Gamma = 0.0999,
        # t = 1), where it drove the measured rate BELOW the bound (0.0904 vs
        # 0.0999) and made the paper's saturation cell look like a violation.
        # Verified against the closed form of Eq. (eq:exactrate): with n_z ~ 0
        # the log-derivative is flat at 2*gamma*kappa^2 across the whole window,
        # while the old expression tracked 2*gamma*kappa^2*exp(-2 g kappa^2 t)
        # to four decimals at every t.
        Ct = complex(np.sum(w * e))
        out.append(float(np.real(slope / Ct)) if abs(Ct) > 1e-12
                   else float("nan"))
    return out, abs(C0)


def stable(g):
    for i in range(len(g) - 1):
        if g[i] > 1e-14 and abs(g[i + 1] - g[i]) / g[i] < 0.15:
            return g[i + 1], ""
    return g[-1], " UNSTABLE"


def pair_ops(code):
    """Hermitian per-pair operators Pi_i - Pi_j over Z-eigenstate codewords."""
    zs = code["z_states"]
    d = len(zs)
    out = []
    for i in range(d):
        for j in range(i + 1, d):
            out.append(((i, j), zs[i] * zs[i].dag() - zs[j] * zs[j].dag()))
    return out


def run_cell(code, meter, paper_val):
    N, d = code["N"], code["d"]
    obs, _ = observables(N, d)
    M = obs[meter]
    ZL = code["logops"][1] if (code["name"].startswith("GKP")
                               or len(code["logops"]) == 2) else code["logops"][2]
    c_ops = code["stab"] + [np.sqrt(GAMMA) * M]
    print(f"{code['name']}  meter={meter}  N={N}  gamma={GAMMA}", flush=True)
    gz, C0 = autocorr_rates_fixed(c_ops, ZL, code["Pcode"], d)
    gs, fl = stable(gz)
    print(f"  Weyl-Z (adjoint-normalized, C0={C0:.4f}):  "
          + "  ".join(f"t={t}: {g:.3e}" for t, g in zip(TSTARS, gz)), flush=True)
    print(f"  -> windowed Gamma_self = {gs:.3e}{fl}", flush=True)
    worst = (None, 0.0)
    for (i, j), Oij in pair_ops(code):
        gp, _ = autocorr_rates_fixed(c_ops, Oij, code["Pcode"], d)
        sp, fp = stable(gp)
        print(f"  pair ({i},{j}) Pi_i-Pi_j: "
              + "  ".join(f"{g:.3e}" for g in gp) + f"  -> {sp:.3e}{fp}",
              flush=True)
        if sp > worst[1]:
            worst = ((i, j), sp)
    print(f"  -> worst-pair Gamma_self = {worst[1]:.3e}  (pair {worst[0]})",
          flush=True)
    dev_w = 100 * (gs - paper_val) / paper_val
    dev_p = 100 * (worst[1] - paper_val) / paper_val
    print(f"  PAPER (spectral triple-run): {paper_val:.1e}   "
          f"deviation: Weyl {dev_w:+.1f}%  worst-pair {dev_p:+.1f}%", flush=True)
    return gs, worst[1]


TABLE2_PRINTED = {
    ("cat", "parity"): 1.0e-1, ("cat", "mod-q"): 1.7e-3, ("cat", "mod-p"): 9.4e-3,
    ("cat", "q"): 1.2e-3, ("cat", "p"): 1.5e-3, ("cat", "n"): 4.8e-3,
    ("GKP", "parity"): 1.4e-8, ("GKP", "mod-q"): 1.7e-3, ("GKP", "mod-p"): 7.4e-2,
    ("GKP", "q"): 3.1e-3, ("GKP", "p"): 5.0e-2, ("GKP", "n"): 1.1e-1}
# GKP-p and GKP-n updated 2026-08-20 from 4.7e-2 and 9.5e-2. Those were the
# pre-fix windowed values: only four cells are fast enough for the C(0)-vs-C(t)
# bug to bite (Gamma*t > 0.05), and of those, cat/parity and GKP/mod-p had been
# computed by an independent route while these two had not. The manuscript
# carries the corrected values; this dict tracks the manuscript.


def table2():
    """Regenerate all twelve d=2 (code, meter) cells of Table II (tab:bound).

    Each row is checked against the printed value as it is produced. Note that
    code["logops"] is [X, Y, Z] for cats but [X, Z, X*Z] for GKP, so the index of
    the logical Z is carried per code rather than inferred from len(logops).
    """
    from perception_frontier import cat_code, gkp_code
    codes = [("cat", cat_code(np.sqrt(1.9), 2), 2), ("GKP", gkp_code(2, 80), 1)]
    meters = ["parity", "mod-q", "mod-p", "q", "p", "n"]
    print(f"TABLE II -- twelve d=2 cells, gamma={GAMMA}", flush=True)
    print(f"{'code':<5s}{'meter':>8s}{'Gamma_self':>13s}{'printed':>11s}"
          f"{'ratio':>8s}{'flag':>11s}", flush=True)
    worst = 0.0
    for short, code, zli in codes:
        N, ZL = code["N"], code["logops"][zli]
        obs, _ = observables(N, 2)
        for meter in meters:
            c_ops = code["stab"] + [np.sqrt(GAMMA) * obs[meter]]
            g, _ = autocorr_rates_fixed(c_ops, ZL, code["Pcode"], 2)
            s, flag = stable(g)
            pr = TABLE2_PRINTED[(short, meter)]
            print(f"{short:<5s}{meter:>8s}{s:>13.4e}{pr:>11.1e}{s/pr:>7.2f}x"
                  f"{(flag.strip() or '-'):>11s}", flush=True)
            worst = max(worst, abs(s / pr - 1))
    print("")
    print(f"largest deviation from the printed table: {100*worst:.0f}%",
          flush=True)
    print("Expect all twelve within roughly 6% of the printed values. GKP-q is "
          "the softest: its windows are still falling at t=4, so stable() "
          "returns the last one; the printed 3.1e-3 agrees with the spectral "
          "value to 0.7%.", flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if mode == "diag":
        for d in (3, 4):
            c = cat_code(np.sqrt(1.9), d)
            ZL = c["logops"][1]
            C0o = complex((ZL * (ZL * c["Pcode"] / d)).tr())
            C0f = complex((ZL.dag() * (ZL * c["Pcode"] / d)).tr())
            print(f"cat d={d}: C0_original={C0o:.3e}  C0_fixed={C0f:.6f}  "
                  f"||Z-Zdag||={(ZL-ZL.dag()).norm():.2f}", flush=True)
    elif mode == "cat3":
        run_cell(cat_code(np.sqrt(1.9), 3), "p", 6.4e-3)
    elif mode == "cat4":
        run_cell(cat_code(np.sqrt(1.9), 4), "q", 1.3e-2)
    elif mode == "cat4lo":
        c = cat_code(np.sqrt(1.9), 4)
        import qudlab.codes as codes
        # rebuild at reduced cutoff for the truncation-convergence check
        Nlo = 48
        from perception_frontier import KD
        a = qt.destroy(Nlo)
        from qudlab.codes import (coherent_legs, _orthonormal_span,
                                  weyl_operators, code_projector)
        al = np.sqrt(1.9)
        z_st = _orthonormal_span(coherent_legs(Nlo, al, 4))
        W = weyl_operators(Nlo, al, 4)
        code = {"name": f"cat d=4 N{Nlo}", "N": Nlo, "d": 4,
                "z_states": z_st, "stab": [np.sqrt(KD) * (a**4 - al**4)],
                "logops": [W["X"], W["Z"]],
                "Pcode": code_projector(Nlo, al, 4)}
        run_cell(code, "q", 1.3e-2)
    elif mode == "table2":
        table2()
    elif mode == "gkp3":
        N = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        g = gkp_code(3, N)
        print(f"GKP d=3 N={N} kernel_ok={g.get('kernel_ok')}", flush=True)
        if not g.get("kernel_ok"):
            print("kernel dirty -- raise N", flush=True)
            return
        run_cell(g, "mod-q", 2.1e-3)


if __name__ == "__main__":
    main()
