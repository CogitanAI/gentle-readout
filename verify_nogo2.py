"""Final verification round.

A. bin1 + 2-harmonic modular: is the exact zero real (V >= 0.02) and N-stable?
B. cat LEG basis + 2-harmonic sin combo: predicted exact zero via
   Lambda01 = -c1^2 A(2l1)/2 - c2^2 A(2l2)/2 + c1 c2 [A(l1-l2) - A(l1+l2)],
   kappa == 0 on the sin lines by symmetry. Solve and confirm.
C. cat4 + 2-harmonic modular attempt.
D. quad class with ||c||=1 and V >= 0.02 (honest quad column, all codes).
E. GKP Delta=0.25 at N=240.
"""
import json
import numpy as np
from scipy.optimize import minimize, brentq, fsolve
import nogo_sweep as ns

def signed_data(cell, params):
    """Signed (M01, L01, Dm, V, trL) for a modular meter."""
    f = np.zeros_like(cell.x)
    for (amp, lam, phi) in params:
        f = f + amp * np.cos(lam * cell.x + phi)
    f0, f1 = f * cell.u0, f * cell.u1
    M00 = np.vdot(cell.u0, f0).real
    M11 = np.vdot(cell.u1, f1).real
    M01 = np.vdot(cell.u0, f1)
    W01 = np.vdot(f0, f1)
    W00 = np.vdot(f0, f0).real
    W11 = np.vdot(f1, f1).real
    L01 = W01 - (M00 * M01 + M01 * M11)
    trL = W00 + W11 - (M00 ** 2 + abs(M01) ** 2) - (M11 ** 2 + abs(M01) ** 2)
    Dm = M00 - M11
    V = 0.25 * Dm ** 2 + abs(M01) ** 2
    return M01, L01, Dm, V, trL

OUT = {}
VMIN = 0.02

def signed_multi(cell, params):
    return signed_data(cell, params)

def B_of(cell, params):
    M01, L01, Dm, V, trL = signed_data(cell, params)
    if V < 1e-300:
        return None
    eps2 = abs(L01) ** 2 / trL if trL > 1e-16 else 0.0
    return dict(B=max(abs(M01) ** 2, eps2) / V, kap2=abs(M01) ** 2 / V,
                eps2=eps2 / V, S=Dm ** 2 / V, Dm=Dm, V=V)

def opt_mod2_vfloor(cell, seeds, nstart=40, rng=np.random.default_rng(7)):
    best = None
    for _ in range(nstart):
        if seeds and rng.random() < 0.5:
            s = seeds[rng.integers(len(seeds))]
            x0 = np.array(s) + rng.normal(scale=0.1, size=6)
        else:
            x0 = np.array([1.0, rng.uniform(0.1, 4), rng.uniform(0, np.pi),
                           rng.uniform(-1, 1), rng.uniform(0.1, 4),
                           rng.uniform(0, np.pi)])
        def f(x):
            # normalize amplitudes so V-floor is meaningful for bounded meters
            nrm = np.hypot(x[0], x[3])
            if nrm < 1e-12:
                return 60.0
            p = [(x[0] / nrm, x[1], x[2]), (x[3] / nrm, x[4], x[5])]
            r = B_of(cell, p)
            if r is None:
                return 60.0
            pen = 1e4 * max(0.0, ns.SIG_MIN - r['S']) ** 2 \
                + 1e6 * max(0.0, VMIN - r['V']) ** 2
            return np.log10(r['B'] + 1e-20) + pen
        r = minimize(f, x0, method='Nelder-Mead',
                     options=dict(maxiter=8000, fatol=1e-14, xatol=1e-12))
        nrm = np.hypot(r.x[0], r.x[3])
        p = [(r.x[0] / nrm, r.x[1], r.x[2]), (r.x[3] / nrm, r.x[4], r.x[5])]
        res = B_of(cell, p)
        if res is None or res['S'] < ns.SIG_MIN - 1e-9 or res['V'] < VMIN * 0.99:
            continue
        if best is None or res['B'] < best[0]['B']:
            best = (res, p)
    return best

N = 120
md = ns.mode(N)

# ------------------------------------------------------------------ A
print("== A: bin1 + 2-harmonic modular, V-floored ==", flush=True)
v0, v1 = ns.build_code('bin1', N)
cell = ns.ModCell(md, v0, v1, 'q')
out = opt_mod2_vfloor(cell, seeds=[[1.0, 2.75, 0.71, 0.3, 1.5, 1.0]])
if out:
    res, p = out
    print(f"bin1 mod2: B={res['B']:.3e} kap2={res['kap2']:.2e} "
          f"eps2={res['eps2']:.2e} V={res['V']:.4f} Dm={res['Dm']:.4f} "
          f"params={[(round(a,5), round(l,5), round(ph,5)) for a,l,ph in p]}",
          flush=True)
    # N-stability
    md2 = ns.mode(160)
    v0b, v1b = ns.build_code('bin1', 160)
    r2 = B_of(ns.ModCell(md2, v0b, v1b, 'q'), p)
    print(f"bin1 mod2 at N=160 (same params): B={r2['B']:.3e}", flush=True)
    OUT['bin1_mod2'] = dict(B=res['B'], V=res['V'], Dm=res['Dm'],
                            B_N160=r2['B'], params=[list(x) for x in p])
else:
    print("bin1 mod2: infeasible", flush=True)
    OUT['bin1_mod2'] = None

# ------------------------------------------------------------------ B
print("\n== B: cat LEG + 2-harmonic sin: predicted exact zero ==", flush=True)
OUT['cat_sin2'] = []
for a2 in [1.0, 2.5, 4.0, 6.0]:
    alpha = np.sqrt(a2)
    v0, v1 = ns.cat2_leg(N, alpha)
    cellc = ns.ModCell(md, v0, v1, 'q')
    l1, l2 = 0.6, 1.8
    def L01_of_t(t):
        nrm = np.hypot(1.0, t)
        p = [(1.0 / nrm, l1, np.pi / 2), (t / nrm, l2, np.pi / 2)]
        return signed_data(cellc, p)[1].real
    ts = np.linspace(-3, 3, 601)
    vals = [L01_of_t(t) for t in ts]
    roots = []
    for i in range(len(ts) - 1):
        if vals[i] * vals[i + 1] < 0:
            roots.append(brentq(L01_of_t, ts[i], ts[i + 1], xtol=1e-15))
    row = dict(a2=a2, nroots=len(roots), sols=[])
    for t in roots:
        nrm = np.hypot(1.0, t)
        p = [(1.0 / nrm, l1, np.pi / 2), (t / nrm, l2, np.pi / 2)]
        r = B_of(cellc, p)
        # N-stability: re-root at N=180
        md2 = ns.mode(180)
        v0b, v1b = ns.cat2_leg(180, alpha)
        cellb = ns.ModCell(md2, v0b, v1b, 'q')
        def L01b(t_):
            nb = np.hypot(1.0, t_)
            return signed_data(cellb, [(1.0 / nb, l1, np.pi / 2),
                                       (t_ / nb, l2, np.pi / 2)])[1].real
        try:
            tb = brentq(L01b, t - 0.05, t + 0.05, xtol=1e-15)
            drift = abs(tb - t)
        except ValueError:
            drift = None
        row['sols'].append(dict(t=t, B=r['B'], kap2=r['kap2'],
                                eps2=r['eps2'], V=r['V'], Dm=r['Dm'],
                                t_drift_N=drift))
        print(f"a2={a2}: t*={t:+.6f}  B={r['B']:.2e} kap2={r['kap2']:.2e} "
              f"V={r['V']:.4f} Dm={r['Dm']:+.4f} "
              f"dt(N120->180)={drift if drift is None else f'{drift:.1e}'}",
              flush=True)
    if not roots:
        print(f"a2={a2}: no sign change in t", flush=True)
    OUT['cat_sin2'].append(row)

# ------------------------------------------------------------------ C
print("\n== C: cat4 + 2-harmonic modular ==", flush=True)
v0, v1 = ns.build_code('cat4_3.0', N)
cell4 = ns.ModCell(md, v0, v1, 'q')
out4 = opt_mod2_vfloor(cell4, seeds=[[1.0, 2.46, 2.60, 0.3, 1.2, 1.0]],
                       nstart=60)
if out4:
    res, p = out4
    print(f"cat4 mod2: B={res['B']:.3e} kap2={res['kap2']:.2e} "
          f"eps2={res['eps2']:.2e} V={res['V']:.4f} Dm={res['Dm']:.4f}",
          flush=True)
    OUT['cat4_mod2'] = dict(B=res['B'], V=res['V'], Dm=res['Dm'],
                            params=[list(x) for x in p])
else:
    print("cat4 mod2: infeasible", flush=True)
    OUT['cat4_mod2'] = None

# ------------------------------------------------------------------ D
print("\n== D: quad class, ||c||=1, V>=0.02 ==", flush=True)
OUT['quad_vfloor'] = []
rng = np.random.default_rng(11)
for code in ['cat2leg_2.5', 'cat2pm_2.5', 'cat4_3.0', 'gkp_0.30',
             'gkp_0.30_0.75', 'bin1', 'bin2', 'fock_0_1', 'fock_1_3']:
    v0, v1 = ns.build_code(code, N)
    qc = ns.QuadCell(md, v0, v1)
    best = None
    for _ in range(60):
        c0 = rng.normal(size=5); c0 /= np.linalg.norm(c0)
        def f(c):
            nc = np.linalg.norm(c)
            if nc < 1e-12:
                return 60.0
            r = qc.evaluate(c / nc)
            if r is None:
                return 60.0
            pen = 1e4 * max(0.0, ns.SIG_MIN - r['S']) ** 2 \
                + 1e6 * max(0.0, VMIN - r['V']) ** 2
            return np.log10(r['B'] + 1e-20) + pen
        r = minimize(f, c0, method='Nelder-Mead',
                     options=dict(maxiter=6000, fatol=1e-13, xatol=1e-11))
        c = r.x / np.linalg.norm(r.x)
        res = qc.evaluate(c)
        if res is None or res['S'] < ns.SIG_MIN - 1e-9 or res['V'] < VMIN * 0.99:
            continue
        if best is None or res['B'] < best[0]['B']:
            best = (res, c.tolist())
    if best is None:
        print(f"{code} quad: infeasible under ||c||=1, V>={VMIN}", flush=True)
        OUT['quad_vfloor'].append(dict(code=code, feasible=False))
    else:
        res, c = best
        print(f"{code} quad: B={res['B']:.3e} kap2={res['kap2']:.2e} "
              f"eps2={res['eps2']:.2e} V={res['V']:.4f} Dm={res['Dm']:.4f} "
              f"c={[round(x,4) for x in c]}", flush=True)
        OUT['quad_vfloor'].append(dict(code=code, feasible=True, B=res['B'],
                                       kap2=res['kap2'], eps2=res['eps2'],
                                       V=res['V'], Dm=res['Dm'], c=c))

# ------------------------------------------------------------------ E
print("\n== E: GKP Delta=0.25 at N=240 ==", flush=True)
md3 = ns.mode(240)
v0, v1 = ns.gkp(240, 0.25)
cellg = ns.ModCell(md3, v0, v1, 'q')
for phi in (np.pi / 4, 3 * np.pi / 4):
    r = B_of(cellg, [(1.0, np.sqrt(np.pi), phi)])
    print(f"D=0.25 N=240 (sqrt(pi), {phi/np.pi:.2f}pi): B={r['B']:.3e} "
          f"V={r['V']:.4f} Dm={r['Dm']:.4f}", flush=True)
    OUT.setdefault('gkp025_N240', []).append(dict(phi=phi, **r))

json.dump(OUT, open('verify_nogo2_results.json', 'w'), indent=1, default=float)
print("\n-> verify_nogo2_results.json", flush=True)
