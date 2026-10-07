"""Focused verification of the exact perfect-pair candidates + honest floors.

A. cat leg basis + sin(lam q): kappa == 0 on the whole line (symmetry);
   Lambda01(lam) sign-changing zeros -> exact perfect pairs. brentq + N-stability.
B. GKP + cos(lam q + phi): 2D common zeros of (M01, L01)(lam, phi);
   transversality (winding number) and N-stability.
C. Grid re-run with V-floor (V >= 0.02) for bounded meter classes;
   honest parity feasibility.
D. bin1: analytic blindness of quad class + modular floor (incl 2-harmonic).
E. Scaling fits for the no-zero cells.
"""
import json
import os
import numpy as np
from scipy.optimize import brentq, fsolve, minimize
import nogo_sweep as ns

OUT = {}

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

# ------------------------------------------------------------------ A
print("== A: cat leg + sin(lam q): exact zeros of Lambda01(lam) ==", flush=True)
OUT['cat_sin'] = []
for a2 in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
    alpha = np.sqrt(a2)
    rows = {}
    kmax = None
    for N in (120, 180):
        md = ns.mode(N)
        v0, v1 = ns.cat2_leg(N, alpha)
        cell = ns.ModCell(md, v0, v1, 'q')
        def L01f(lam):
            return signed_data(cell, [(1.0, lam, np.pi / 2)])[1].real
        lams = np.linspace(0.02, 2.5, 400)
        vals = np.array([L01f(l) for l in lams])
        zeros = []
        for i in range(len(lams) - 1):
            if vals[i] * vals[i + 1] < 0:
                lz = brentq(L01f, lams[i], lams[i + 1], xtol=1e-14)
                M01, L01, Dm, V, trL = signed_data(cell, [(1.0, lz, np.pi / 2)])
                if V < 0.02:
                    continue
                B = max(abs(M01) ** 2, abs(L01) ** 2 / max(trL, 1e-300)) / V
                zeros.append(dict(lam=lz, Dm=Dm, V=V, kap=abs(M01),
                                  L01=abs(L01), trL=trL, B=B))
        rows[N] = zeros
        if N == 120:
            kmax = max(abs(signed_data(cell, [(1.0, l, np.pi / 2)])[0])
                       for l in lams[::20])
    pred = np.pi / (4 * np.sqrt(2) * alpha)
    z120, z180 = rows[120], rows[180]
    first = z120[0] if z120 else None
    stab = (abs(z120[0]['lam'] - z180[0]['lam']) if (z120 and z180) else None)
    OUT['cat_sin'].append(dict(a2=a2, nzeros=len(z120), first=first,
                               lam_shift_N=stab, lam_pred_k0=pred,
                               kappa_max_on_line=kmax))
    if first:
        print(f"a2={a2}: {len(z120)} zeros; first lam*={first['lam']:.6f} "
              f"(pred~{pred:.4f}) Dm={first['Dm']:.4f} V={first['V']:.4f} "
              f"B={first['B']:.2e} kap={first['kap']:.2e} "
              f"dlam(N120->180)={stab:.2e} kap_max_line={kmax:.1e}", flush=True)
    else:
        print(f"a2={a2}: no zeros with V>=0.02", flush=True)

# ------------------------------------------------------------------ B
print("\n== B: GKP + cos(lam q + phi): 2D common zeros ==", flush=True)
OUT['gkp_zero'] = []
def gkp_zero_find(Delta, aspect, N, seed):
    md = ns.mode(N)
    v0, v1 = ns.gkp(N, Delta, aspect)
    cell = ns.ModCell(md, v0, v1, 'q')
    def F(x):
        M01, L01, Dm, V, trL = signed_data(cell, [(1.0, x[0], x[1])])
        return [M01.real, L01.real]
    sol, info, ier, msg = fsolve(F, seed, full_output=True, xtol=1e-13)
    M01, L01, Dm, V, trL = signed_data(cell, [(1.0, sol[0], sol[1])])
    B = max(abs(M01) ** 2, abs(L01) ** 2 / max(trL, 1e-300)) / V
    angs = np.unwrap([np.arctan2(*reversed(F([sol[0] + 3e-3 * np.cos(th),
                                              sol[1] + 3e-3 * np.sin(th)])))
                      for th in np.linspace(0, 2 * np.pi, 60)])
    wind = (angs[-1] - angs[0]) / (2 * np.pi)
    return dict(ok=bool(ier == 1), lam=float(sol[0]), phi=float(sol[1]),
                Dm=Dm, V=V, B=B, kap=abs(M01), L01=abs(L01), trL=trL,
                winding=wind)

for Delta in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    seed = [np.sqrt(np.pi), 3 * np.pi / 4]
    rN = {}
    for N in (120, 160, 200):
        rN[N] = gkp_zero_find(Delta, 1.0, N, seed)
        seed = [rN[N]['lam'], rN[N]['phi']]
    r = rN[200]
    dlam = abs(rN[160]['lam'] - rN[200]['lam'])
    OUT['gkp_zero'].append(dict(Delta=Delta, aspect=1.0, dlam_N=dlam, **r))
    print(f"D={Delta}: ok={r['ok']} lam*={r['lam']:.6f} phi*={r['phi']:.6f} "
          f"(phi*/pi={r['phi']/np.pi:.5f}) Dm={r['Dm']:.4f} V={r['V']:.4f} "
          f"B={r['B']:.2e} wind={r['winding']:.2f} dlam(N)={dlam:.1e}",
          flush=True)

for aspect in [0.5, 0.75]:
    seed = [np.sqrt(np.pi / aspect), np.pi / 4]
    rN = {}
    for N in (120, 160, 200):
        rN[N] = gkp_zero_find(0.30, aspect, N, seed)
        seed = [rN[N]['lam'], rN[N]['phi']]
    r = rN[200]
    dlam = abs(rN[160]['lam'] - rN[200]['lam'])
    OUT['gkp_zero'].append(dict(Delta=0.30, aspect=aspect, dlam_N=dlam, **r))
    print(f"D=0.30 r={aspect}: ok={r['ok']} lam*={r['lam']:.6f} "
          f"phi*={r['phi']:.6f} Dm={r['Dm']:.4f} V={r['V']:.4f} "
          f"B={r['B']:.2e} wind={r['winding']:.2f} dlam(N)={dlam:.1e}",
          flush=True)

# ------------------------------------------------------------------ C
print("\n== C: V-floored re-run of bounded-meter cells ==", flush=True)
VMIN = 0.02
def opt_mod_vfloor(cell):
    best = None
    for lam in np.arange(0.05, 6.0, 0.025):
        for phi in np.arange(0, np.pi, np.pi / 48):
            res = cell.evaluate([(1.0, lam, phi)])
            if res is None or res['S'] < ns.SIG_MIN or res['V'] < VMIN:
                continue
            if best is None or res['B'] < best[0]['B']:
                best = (res, [lam, phi])
    if best is None:
        return None
    def f(x):
        r = cell.evaluate([(1.0, x[0], x[1])])
        if r is None:
            return 60.0
        pen = 1e4 * max(0.0, ns.SIG_MIN - r['S']) ** 2 \
            + 1e6 * max(0.0, VMIN - r['V']) ** 2
        return np.log10(r['B'] + 1e-20) + pen
    r = minimize(f, best[1], method='Nelder-Mead',
                 options=dict(maxiter=4000, fatol=1e-13, xatol=1e-12))
    res = cell.evaluate([(1.0, r.x[0], r.x[1])])
    if res and res['S'] >= ns.SIG_MIN - 1e-9 and res['V'] >= VMIN * 0.99:
        return (res, r.x.tolist())
    return best

N = 120
md = ns.mode(N)
OUT['vfloor'] = []
cells = [('cat2leg_2.5', 'modq'), ('cat2leg_2.5', 'modp'),
         ('cat2pm_2.5', 'modq'), ('cat2pm_2.5', 'modp'),
         ('cat4_3.0', 'modq'), ('cat4_3.0', 'modp'),
         ('gkp_0.30', 'modq'), ('gkp_0.30', 'modp'),
         ('gkp_0.30_0.75', 'modq'), ('gkp_0.30_0.75', 'modp'),
         ('bin1', 'modq'), ('bin1', 'modp'),
         ('bin2', 'modq'), ('bin2', 'modp'),
         ('fock_0_1', 'modq'), ('fock_1_3', 'modq')]
for code, cls in cells:
    v0, v1 = ns.build_code(code, N)
    cell = ns.ModCell(md, v0, v1, cls[-1])
    out = opt_mod_vfloor(cell)
    if out is None:
        OUT['vfloor'].append(dict(code=code, cls=cls, feasible=False))
        print(f"{code} {cls}: infeasible under V>={VMIN}", flush=True)
        continue
    res, x = out
    OUT['vfloor'].append(dict(code=code, cls=cls, feasible=True, B=res['B'],
                              kap2=res['kap2'], eps2=res['eps2'], V=res['V'],
                              Dm=res['Dm'], params=x))
    print(f"{code} {cls}: B={res['B']:.3e} kap2={res['kap2']:.2e} "
          f"eps2={res['eps2']:.2e} V={res['V']:.4f} Dm={res['Dm']:.4f} "
          f"lam={x[0]:.4f} phi={x[1]:.4f}", flush=True)

OUT['parity'] = []
for code in ['cat2leg_2.5', 'cat2pm_2.5', 'cat4_3.0', 'gkp_0.30',
             'gkp_0.30_0.75', 'bin1', 'bin2', 'fock_0_1', 'fock_1_3']:
    v0, v1 = ns.build_code(code, N)
    r = ns.parity_eval(md, v0, v1)
    feas = (r is not None) and abs(r['Dm']) > 1e-8
    OUT['parity'].append(dict(code=code, feasible=bool(feas),
                              Dm=(r['Dm'] if r else 0.0),
                              B=(r['B'] if feas else None)))
    print(f"{code} parity: Dm={(r['Dm'] if r else 0.0):.3e} -> "
          f"{('B=%.2e' % r['B']) if feas else 'BLIND'}", flush=True)

# ------------------------------------------------------------------ D
print("\n== D: bin1 analytics + 2-harmonic floor ==", flush=True)
v0, v1 = ns.build_code('bin1', N)
c = ns.QuadCell(md, v0, v1)
dm_quad = [abs((c.Mel[i][0, 0] - c.Mel[i][1, 1]).real) for i in range(5)]
off_quad = [abs(c.Mel[i][0, 1]) for i in range(5)]
print(f"bin1 quad |Dm_i| per generator (q,p,q2,p2,qp): "
      f"{['%.2e' % d for d in dm_quad]}")
print(f"bin1 quad |M01_i|: {['%.2e' % d for d in off_quad]}", flush=True)
OUT['bin1_quad_Dm'] = dm_quad
cellq = ns.ModCell(md, v0, v1, 'q')
seed = [r for r in OUT['vfloor']
        if r.get('code') == 'bin1' and r.get('cls') == 'modq'
        and r.get('feasible')]
if seed:
    s = seed[0]['params']
    b2 = ns.opt_mod2(cellq, s[0], s[1], nstart=30)
    if b2:
        print(f"bin1 modq 2-harmonic: B={b2[0]['B']:.3e} "
              f"(1-harm {seed[0]['B']:.3e})", flush=True)
        OUT['bin1_mod2'] = b2[0]['B']

# ------------------------------------------------------------------ E
print("\n== E: scaling fits (no-zero cells) ==", flush=True)
res = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nogo_results.json')))
sc = res['scaling']
a2 = np.array([r['a2'] for r in sc['cat_leg_quad']])
Bq = np.array([r['B'] for r in sc['cat_leg_quad']], float)
Bp = np.array([r['B'] for r in sc['cat_leg_pureq']], float)
for name, B in [('cat_quadclass', Bq), ('cat_pure_q', Bp)]:
    A = np.column_stack([np.ones_like(a2), a2, np.log(a2)])
    coef, *_ = np.linalg.lstsq(A, np.log(B), rcond=None)
    resid = np.log(B) - A @ coef
    A2 = np.column_stack([np.ones_like(a2), a2])
    c2, *_ = np.linalg.lstsq(A2, np.log(B), rcond=None)
    r2 = 1 - np.var(np.log(B) - A2 @ c2) / np.var(np.log(B))
    print(f"{name}: ln B = {coef[0]:.2f} + ({coef[1]:.3f}) a2 + "
          f"({coef[2]:.2f}) ln a2  [max|res|={abs(resid).max():.3f}] | "
          f"pure-exp slope {c2[1]:.3f} R2={r2:.5f}", flush=True)
    OUT[f'fit_{name}'] = dict(c0=coef[0], c_a2=coef[1], c_ln=coef[2],
                              exp_slope=c2[1], exp_R2=r2)

# ------------------------------------------------------------------ F
print("\n== F: cat leg + modq under V-floor: scaling vs a2 ==", flush=True)
OUT['cat_modq_vfloor'] = []
for a2 in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
    v0, v1 = ns.cat2_leg(N, np.sqrt(a2))
    cell = ns.ModCell(md, v0, v1, 'q')
    out = opt_mod_vfloor(cell)
    if out is None:
        OUT['cat_modq_vfloor'].append(dict(a2=a2, feasible=False))
        print(f"a2={a2}: infeasible", flush=True)
        continue
    res, x = out
    OUT['cat_modq_vfloor'].append(dict(a2=a2, feasible=True, B=res['B'],
                                       kap2=res['kap2'], eps2=res['eps2'],
                                       V=res['V'], lam=x[0], phi=x[1]))
    print(f"a2={a2}: B={res['B']:.3e} kap2={res['kap2']:.2e} "
          f"eps2={res['eps2']:.2e} V={res['V']:.4f} lam={x[0]:.4f} "
          f"phi={x[1]:.4f}", flush=True)
a2s = np.array([r['a2'] for r in OUT['cat_modq_vfloor'] if r['feasible']])
Bs = np.array([r['B'] for r in OUT['cat_modq_vfloor'] if r['feasible']])
if len(a2s) > 3:
    A2m = np.column_stack([np.ones_like(a2s), a2s])
    cm, *_ = np.linalg.lstsq(A2m, np.log(Bs), rcond=None)
    r2m = 1 - np.var(np.log(Bs) - A2m @ cm) / np.var(np.log(Bs))
    print(f"cat modq(Vfloor) fit: exp slope {cm[1]:.3f} R2={r2m:.5f}",
          flush=True)
    OUT['fit_cat_modq_vfloor'] = dict(exp_slope=cm[1], exp_R2=r2m, c0=cm[0])

# ------------------------------------------------------------------ G
print("\n== G: GKP mechanism checks ==", flush=True)
v0, v1 = ns.gkp(N, 0.35)
cell = ns.ModCell(md, v0, v1, 'q')
sq = np.sqrt(np.pi)
kaps, L01s = [], []
for phi in np.linspace(0, np.pi, 9):
    M01, L01, Dm, V, trL = signed_data(cell, [(1.0, sq, phi)])
    kaps.append(abs(M01)); L01s.append(L01.real)
    print(f"  phi={phi:.4f}: kappa={abs(M01):.3e}  L01={L01.real:+.6e} "
          f"( -C*cos2phi pred: {-L01s[0]*np.cos(2*phi):+.6e} )", flush=True)
OUT['gkp_mech'] = dict(phis=list(np.linspace(0, np.pi, 9)), kaps=kaps,
                       L01s=L01s)
print(f"  max kappa on lam=sqrt(pi) line: {max(kaps):.2e}  "
      f"(exact-zero mechanism iff ~1e-16)", flush=True)
# second zero at phi=pi/4
M01, L01, Dm, V, trL = signed_data(cell, [(1.0, sq, np.pi / 4)])
B = max(abs(M01) ** 2, abs(L01) ** 2 / max(trL, 1e-300)) / V
print(f"  phi=pi/4 zero: B={B:.2e} Dm={Dm:.4f} V={V:.4f}", flush=True)
OUT['gkp_pi4'] = dict(B=B, Dm=Dm, V=V)

# bin2 transversality
v0, v1 = ns.build_code('bin2', N)
cellb = ns.ModCell(md, v0, v1, 'q')
def Fb(x):
    M01, L01, *_ = signed_data(cellb, [(1.0, x[0], x[1])])
    return [M01.real, L01.real]
solb, infob, ierb, _ = fsolve(Fb, [2.256, 3 * np.pi / 4], full_output=True,
                              xtol=1e-13)
angs = np.unwrap([np.arctan2(*reversed(Fb([solb[0] + 3e-3 * np.cos(th),
                                           solb[1] + 3e-3 * np.sin(th)])))
                  for th in np.linspace(0, 2 * np.pi, 60)])
windb = (angs[-1] - angs[0]) / (2 * np.pi)
M01, L01, Dm, V, trL = signed_data(cellb, [(1.0, solb[0], solb[1])])
Bb = max(abs(M01) ** 2, abs(L01) ** 2 / max(trL, 1e-300)) / V
print(f"bin2 zero: lam*={solb[0]:.8f} phi*={solb[1]:.8f} "
      f"(phi/pi={solb[1]/np.pi:.6f}) B={Bb:.2e} Dm={Dm:.4f} V={V:.4f} "
      f"winding={windb:.2f}", flush=True)
OUT['bin2_zero'] = dict(lam=float(solb[0]), phi=float(solb[1]), B=Bb,
                        Dm=Dm, V=V, winding=windb, ok=bool(ierb == 1))

json.dump(OUT, open('verify_nogo_results.json', 'w'), indent=1, default=float)
print("\n-> verify_nogo_results.json", flush=True)
