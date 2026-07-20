"""No-go sweep: does any physically accessible single-mode meter achieve
PERFECT self-reporting readout (kappa = 0 AND Lambda01 = 0, Delta_m != 0)
for any bosonic code?

Definitions (as in readability.tex):
  P     = |0><0| + |1><1|   (orthonormal codewords, fixed read basis)
  PMP   = mbar P + (Dm/2) Z_P + kappa X_P,  Dm = M00 - M11, kappa = |M01|
  E     = Pperp M P,  Lambda = P M Pperp M P,  eps_KL^2 = |L01|^2 / tr(Lambda)
  V     = (Dm/2)^2 + kappa^2   (within-code meter variance)
  Badness  B = max(kappa^2, eps_KL^2) / V     (scale-invariant; B is
           degree-2 homogeneous in M and V restores invariance)
  Signal constraint  S = Dm^2 / V >= 0.5.

Menu classes:
  (i)   quad : span{q, p, q^2, p^2, qp+pq}   (identity is irrelevant: it
               drops from Dm, kappa AND from E)
  (ii)  parity: exp(i pi n)  (single element)
  (iii) modq : cos(lam*q + phi), lam, phi tunable (covers sin via phi)
        modp : cos(lam*p + phi)
  (iv)  linear combos within a class: quad already closed; a 2-harmonic
        modular check is run on the best cells.

Convention: q = (a + a^dag)/sqrt(2).
"""
from __future__ import annotations
import json, sys, time
import numpy as np
from scipy.optimize import minimize

RNG = np.random.default_rng(20260719)
FLOOR = 1e-20          # log-floor for the objective
ZERO_TOL = 1e-14       # below this, B counts as numerically zero
SIG_MIN = 0.5          # Dm^2/V constraint

# ---------------------------------------------------------------- operators
def ops(N):
    a = np.diag(np.sqrt(np.arange(1, N)), 1)
    ad = a.conj().T
    q = (a + ad) / np.sqrt(2)
    p = 1j * (ad - a) / np.sqrt(2)
    n = ad @ a
    return a, ad, q, p, n

class Mode:
    def __init__(self, N):
        self.N = N
        self.a, self.ad, self.q, self.p, self.n = ops(N)
        self.q2 = self.q @ self.q
        self.p2 = self.p @ self.p
        self.qp = self.q @ self.p + self.p @ self.q
        self.parity = np.diag((-1.0) ** np.arange(N)).astype(complex)
        # eigendecompositions for fast cos(lam*x + phi)
        self.xq, self.Uq = np.linalg.eigh(self.q)
        self.xp, self.Up = np.linalg.eigh(self.p)

_MODES = {}
def mode(N):
    if N not in _MODES:
        _MODES[N] = Mode(N)
    return _MODES[N]

# ---------------------------------------------------------------- codes
def lowdin(vs):
    """Symmetric orthonormalization of a list of N-vectors."""
    V = np.column_stack(vs)
    G = V.conj().T @ V
    w, U = np.linalg.eigh(G)
    if w.min() < 1e-13:
        raise ValueError(f"Gram nearly singular: {w}")
    Gm = U @ np.diag(w ** -0.5) @ U.conj().T
    O = V @ Gm
    return [O[:, j].copy() for j in range(O.shape[1])]

def coherent(N, alpha):
    ns = np.arange(N)
    from scipy.special import gammaln
    logc = -0.5 * abs(alpha) ** 2 + ns * np.log(alpha + 0j) - 0.5 * gammaln(ns + 1)
    return np.exp(logc)

def cat2_leg(N, alpha):
    """Two-legged cat, LEG (protected-variable) read basis."""
    v0, v1 = lowdin([coherent(N, alpha), coherent(N, -alpha)])
    return v0, v1

def cat2_pm(N, alpha):
    """Two-legged cat, even/odd (parity) read basis."""
    cp = coherent(N, alpha) + coherent(N, -alpha)
    cm = coherent(N, alpha) - coherent(N, -alpha)
    return cp / np.linalg.norm(cp), cm / np.linalg.norm(cm)

def cat4(N, alpha):
    """Four-legged cat qubit: codewords in n=0 mod 4 and n=2 mod 4 sectors."""
    legs = [coherent(N, alpha * 1j ** k) for k in range(4)]
    v0 = sum(legs)
    v1 = legs[0] - legs[1] + legs[2] - legs[3]
    return v0 / np.linalg.norm(v0), v1 / np.linalg.norm(v1)

def squeezed_vac(N, r):
    """Squeezed vacuum with position variance reduced: Var q = e^{-2r}/2."""
    # S = exp(r/2 (a^2 - ad^2)) acting on vacuum, Fock recursion:
    # c_{2m} ~ (tanh r)^m sqrt((2m)!)/(2^m m!) up to norm, with sign for q-squeeze
    ns = np.arange(N)
    c = np.zeros(N, dtype=complex)
    t = np.tanh(r)
    from scipy.special import gammaln
    m = np.arange(0, (N + 1) // 2)
    logamp = 0.5 * gammaln(2 * m + 1) - m * np.log(2.0) - gammaln(m + 1) \
             + m * np.log(abs(t) + 1e-300)
    amp = ((-np.sign(t)) ** m) * np.exp(logamp)   # sign => q squeezed for r>0
    c[2 * m] = amp
    c /= np.linalg.norm(c)
    return c

def displace_q(N, x0, vec):
    """Shift q by x0: D(x0/sqrt(2)) with real argument."""
    md = mode(N)
    # exp(beta ad - beta a) with beta real = exp(-i x0 p)... use p eigenbasis
    f = np.exp(-1j * x0 * md.xp)
    return md.Up @ (f * (md.Up.conj().T @ vec))

def gkp(N, Delta, aspect=1.0):
    """Finite-energy square/rectangular GKP.
    Peaks of logical mu at q = (2s+mu) * lq, lq = sqrt(pi*aspect);
    each peak a q-squeezed vacuum with sigma_q = Delta/sqrt(2)*? (Var q = Delta^2/2),
    Gaussian envelope exp(-Delta^2 x^2 / 2)."""
    lq = np.sqrt(np.pi * aspect)
    r = -np.log(Delta)
    sq = squeezed_vac(N, r)
    out = []
    for mu in (0, 1):
        v = np.zeros(N, dtype=complex)
        smax = int(np.ceil(6.0 / (Delta * lq))) + 2
        for s in range(-smax, smax + 1):
            x = (2 * s + mu) * lq
            w = np.exp(-0.5 * Delta ** 2 * x ** 2)
            if w < 1e-14:
                continue
            v = v + w * displace_q(N, x, sq)
        out.append(v / np.linalg.norm(v))
    return lowdin(out)

def binomial1(N):
    v0 = np.zeros(N, complex); v0[0] = 1 / np.sqrt(2); v0[4] = 1 / np.sqrt(2)
    v1 = np.zeros(N, complex); v1[2] = 1.0
    return v0, v1

def binomial2(N):
    v0 = np.zeros(N, complex); v0[0] = 0.5; v0[4] = np.sqrt(3) / 2
    v1 = np.zeros(N, complex); v1[2] = np.sqrt(3) / 2; v1[6] = 0.5
    return v0, v1

def fock(N, n0, n1):
    v0 = np.zeros(N, complex); v0[n0] = 1
    v1 = np.zeros(N, complex); v1[n1] = 1
    return v0, v1

# ---------------------------------------------------------------- badness
def badness_from_data(M00, M11, M01, w0, w1):
    """w_mu = M v_mu (full vectors). Returns dict of diagnostics."""
    Dm = (M00 - M11).real
    kap2 = abs(M01) ** 2
    V = 0.25 * Dm ** 2 + kap2
    if V < 1e-300:
        return None
    # Lambda_{mu nu} = <w_mu|w_nu> - sum_sigma M_{mu sigma} M_{sigma nu}
    Mm = np.array([[M00, M01], [np.conj(M01), M11]])
    W = np.array([[np.vdot(w0, w0), np.vdot(w0, w1)],
                  [np.vdot(w1, w0), np.vdot(w1, w1)]])
    Lam = W - Mm @ Mm
    trL = max(Lam[0, 0].real + Lam[1, 1].real, 0.0)
    L01 = Lam[0, 1]
    eps2 = (abs(L01) ** 2 / trL) if trL > 1e-16 * max(abs(W).max(), 1e-300) else 0.0
    eps2 = max(eps2, 0.0)
    B = max(kap2, eps2) / V
    S = Dm ** 2 / V
    return dict(B=B, kap2=kap2 / V, eps2=eps2 / V, S=S, Dm=Dm, V=V,
                trL=trL, kap2_raw=kap2, eps2_raw=eps2)

def eval_quad(md, v0, v1, c):
    Os = [md.q, md.p, md.q2, md.p2, md.qp]
    # precomputation shortcut handled by caller cache
    raise RuntimeError("use QuadCell")

class QuadCell:
    """Cache O_i v_mu and (O_i)_{mu nu} for fast quadratic-class evaluation."""
    def __init__(self, md, v0, v1):
        Os = [md.q, md.p, md.q2, md.p2, md.qp]
        self.Ov = [[O @ v0, O @ v1] for O in Os]
        self.Mel = [np.array([[np.vdot(v0, Ov0), np.vdot(v0, Ov1)],
                              [np.vdot(v1, Ov0), np.vdot(v1, Ov1)]])
                    for (Ov0, Ov1) in self.Ov]
        self.k = len(Os)

    def evaluate(self, c):
        Mm = sum(c[i] * self.Mel[i] for i in range(self.k))
        w0 = sum(c[i] * self.Ov[i][0] for i in range(self.k))
        w1 = sum(c[i] * self.Ov[i][1] for i in range(self.k))
        return badness_from_data(Mm[0, 0], Mm[1, 1], Mm[0, 1], w0, w1)

class ModCell:
    """cos(lam*x + phi) meters in the q or p eigenbasis."""
    def __init__(self, md, v0, v1, which='q'):
        if which == 'q':
            self.x, U = md.xq, md.Uq
        else:
            self.x, U = md.xp, md.Up
        self.u0 = U.conj().T @ v0
        self.u1 = U.conj().T @ v1

    def evaluate(self, params):
        # params: [(amp, lam, phi), ...] summed
        f = np.zeros_like(self.x)
        for (amp, lam, phi) in params:
            f = f + amp * np.cos(lam * self.x + phi)
        f0, f1 = f * self.u0, f * self.u1
        M00 = np.vdot(self.u0, f0); M11 = np.vdot(self.u1, f1)
        M01 = np.vdot(self.u0, f1)
        return badness_from_data(M00, M11, M01, f0, f1)

def parity_eval(md, v0, v1):
    w0 = md.parity @ v0; w1 = md.parity @ v1
    return badness_from_data(np.vdot(v0, w0), np.vdot(v1, w1),
                             np.vdot(v0, w1), w0, w1)

# ---------------------------------------------------------------- optimizers
def penalized(res):
    if res is None:
        return 60.0
    pen = 1e4 * max(0.0, SIG_MIN - res['S']) ** 2
    return np.log10(res['B'] + FLOOR) + pen

def opt_quad(cell, nstart=60):
    best = None
    for _ in range(nstart):
        c0 = RNG.normal(size=5)
        c0 /= np.linalg.norm(c0)
        r = minimize(lambda c: penalized(cell.evaluate(c)), c0,
                     method='Nelder-Mead',
                     options=dict(maxiter=4000, fatol=1e-12, xatol=1e-10))
        res = cell.evaluate(r.x)
        if res is None or res['S'] < SIG_MIN - 1e-9:
            continue
        if best is None or res['B'] < best[0]['B']:
            best = (res, r.x.tolist())
    return best

def opt_mod(cell, lam_max=6.0):
    lams = np.arange(0.05, lam_max, 0.05)
    phis = np.arange(0, np.pi, np.pi / 24)
    cands = []
    for lam in lams:
        for phi in phis:
            res = cell.evaluate([(1.0, lam, phi)])
            if res is None or res['S'] < SIG_MIN:
                continue
            cands.append((res['B'], lam, phi))
    if not cands:
        return None
    cands.sort()
    best = None
    for (_, lam, phi) in cands[:12]:
        r = minimize(lambda x: penalized(cell.evaluate([(1.0, x[0], x[1])])),
                     [lam, phi], method='Nelder-Mead',
                     options=dict(maxiter=2000, fatol=1e-12, xatol=1e-11))
        res = cell.evaluate([(1.0, r.x[0], r.x[1])])
        if res is None or res['S'] < SIG_MIN - 1e-9:
            continue
        if best is None or res['B'] < best[0]['B']:
            best = (res, r.x.tolist())
    return best

def opt_mod2(cell, seed_lam, seed_phi, nstart=25):
    """Two-harmonic modular combo: c1 cos(l1 x+f1) + c2 cos(l2 x+f2)."""
    best = None
    for _ in range(nstart):
        x0 = np.array([1.0, seed_lam, seed_phi,
                       RNG.uniform(0.0, 0.5), RNG.uniform(0.1, 6.0),
                       RNG.uniform(0, np.pi)])
        def f(x):
            return penalized(cell.evaluate([(x[0], x[1], x[2]),
                                            (x[3], x[4], x[5])]))
        r = minimize(f, x0, method='Nelder-Mead',
                     options=dict(maxiter=6000, fatol=1e-12, xatol=1e-11))
        res = cell.evaluate([(r.x[0], r.x[1], r.x[2]), (r.x[3], r.x[4], r.x[5])])
        if res is None or res['S'] < SIG_MIN - 1e-9:
            continue
        if best is None or res['B'] < best[0]['B']:
            best = (res, r.x.tolist())
    return best

# ---------------------------------------------------------------- sweep
def build_code(name, N):
    if name.startswith('cat2leg'):
        return cat2_leg(N, np.sqrt(float(name.split('_')[1])))
    if name.startswith('cat2pm'):
        return cat2_pm(N, np.sqrt(float(name.split('_')[1])))
    if name.startswith('cat4'):
        return cat4(N, np.sqrt(float(name.split('_')[1])))
    if name.startswith('gkp'):
        parts = name.split('_')
        Delta = float(parts[1]); aspect = float(parts[2]) if len(parts) > 2 else 1.0
        return gkp(N, Delta, aspect)
    if name == 'bin1':
        return binomial1(N)
    if name == 'bin2':
        return binomial2(N)
    if name.startswith('fock'):
        _, n0, n1 = name.split('_')
        return fock(N, int(n0), int(n1))
    raise ValueError(name)

def run_cell(code, N, cls):
    md = mode(N)
    v0, v1 = build_code(code, N)
    if cls == 'quad':
        out = opt_quad(QuadCell(md, v0, v1))
    elif cls == 'parity':
        res = parity_eval(md, v0, v1)
        out = (res, None) if (res and res['S'] >= SIG_MIN) else None
        if out is None and res is not None:
            return dict(code=code, cls=cls, feasible=False, S_best=res['S'])
    elif cls == 'modq':
        out = opt_mod(ModCell(md, v0, v1, 'q'))
    elif cls == 'modp':
        out = opt_mod(ModCell(md, v0, v1, 'p'))
    else:
        raise ValueError(cls)
    if out is None:
        return dict(code=code, cls=cls, feasible=False)
    res, x = out
    return dict(code=code, cls=cls, feasible=True, B=res['B'],
                kap2=res['kap2'], eps2=res['eps2'], S=res['S'],
                Dm=res['Dm'], V=res['V'], params=x)

def recheck(code, Nbig, cls, params):
    """Re-evaluate a found optimum at larger cutoff."""
    md = mode(Nbig)
    v0, v1 = build_code(code, Nbig)
    if cls == 'quad':
        res = QuadCell(md, v0, v1).evaluate(np.array(params))
    elif cls == 'parity':
        res = parity_eval(md, v0, v1)
    elif cls in ('modq', 'modp'):
        res = ModCell(md, v0, v1, cls[-1]).evaluate([(1.0, params[0], params[1])])
    return res['B'] if res else None

def main():
    t0 = time.time()
    N, Nbig = 120, 160
    codes = ['cat2leg_2.5', 'cat2pm_2.5', 'cat4_3.0',
             'gkp_0.30', 'gkp_0.30_0.75',
             'bin1', 'bin2', 'fock_0_1', 'fock_1_3']
    classes = ['quad', 'parity', 'modq', 'modp']
    results = []
    for code in codes:
        for cls in classes:
            r = run_cell(code, N, cls)
            if r.get('feasible') and r.get('params') is not None or \
               (r.get('feasible') and cls == 'parity'):
                r['B_Nbig'] = recheck(code, Nbig, cls,
                                      r.get('params') or [])
            results.append(r)
            print(json.dumps(r), flush=True)
    print(f"# sweep done in {time.time()-t0:.0f}s", flush=True)

    # ---------------- Task B: scaling ----------------
    scal = {'cat_leg_quad': [], 'cat_leg_pureq': [], 'gkp_mod': [],
            'cat_pm_parity': []}
    for a2 in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
        md = mode(N)
        v0, v1 = cat2_leg(N, np.sqrt(a2))
        cell = QuadCell(md, v0, v1)
        out = opt_quad(cell, nstart=40)
        rq = cell.evaluate(np.array([1.0, 0, 0, 0, 0]))  # pure q
        Bopt = out[0]['B'] if out else None
        Bbig = None
        if out:
            mdb = mode(Nbig)
            v0b, v1b = cat2_leg(Nbig, np.sqrt(a2))
            Bbig = QuadCell(mdb, v0b, v1b).evaluate(np.array(out[1]))['B']
        scal['cat_leg_quad'].append(dict(a2=a2, B=Bopt, B_Nbig=Bbig,
                                         kap2=out[0]['kap2'] if out else None,
                                         eps2=out[0]['eps2'] if out else None))
        scal['cat_leg_pureq'].append(dict(a2=a2, B=rq['B'], kap2=rq['kap2'],
                                          eps2=rq['eps2'], S=rq['S']))
        # parity on the +- basis (exact-zero check)
        w0, w1 = cat2_pm(N, np.sqrt(a2))
        rp = parity_eval(md, w0, w1)
        scal['cat_pm_parity'].append(dict(a2=a2, B=rp['B'], kap2=rp['kap2'],
                                          eps2=rp['eps2'], S=rp['S'],
                                          trL=rp['trL']))
        print(json.dumps(dict(scaling='cat', a2=a2,
                              B_quad=Bopt, B_pureq=rq['B'],
                              B_parity_pm=rp['B'])), flush=True)
    for Delta in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        md = mode(N)
        v0, v1 = gkp(N, Delta)
        cell = ModCell(md, v0, v1, 'q')
        out = opt_mod(cell)
        Bbig = None
        if out:
            mdb = mode(Nbig)
            v0b, v1b = gkp(Nbig, Delta)
            Bbig = ModCell(mdb, v0b, v1b, 'q').evaluate(
                [(1.0, out[1][0], out[1][1])])['B']
        scal['gkp_mod'].append(dict(Delta=Delta,
                                    B=out[0]['B'] if out else None,
                                    B_Nbig=Bbig,
                                    lam=out[1][0] if out else None,
                                    phi=out[1][1] if out else None,
                                    kap2=out[0]['kap2'] if out else None,
                                    eps2=out[0]['eps2'] if out else None))
        print(json.dumps(dict(scaling='gkp', Delta=Delta,
                              B=out[0]['B'] if out else None,
                              B_Nbig=Bbig)), flush=True)

    # two-harmonic modular on best GKP cell
    md = mode(N)
    v0, v1 = gkp(N, 0.30)
    cell = ModCell(md, v0, v1, 'q')
    seed = opt_mod(cell)
    two = opt_mod2(cell, seed[1][0], seed[1][1]) if seed else None
    extra = dict(gkp_mod2harm=dict(B1=seed[0]['B'] if seed else None,
                                   B2=two[0]['B'] if two else None,
                                   params=two[1] if two else None))
    print(json.dumps(extra), flush=True)

    with open(OUT, 'w') as f:
        json.dump(dict(grid=results, scaling=scal, extra=extra), f, indent=1)
    print(f"# total {time.time()-t0:.0f}s -> {OUT}", flush=True)

OUT = sys.argv[1] if len(sys.argv) > 1 else 'nogo_results.json'
if __name__ == '__main__':
    main()
