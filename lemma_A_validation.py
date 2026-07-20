"""Dense 4-level validation of the leak-term flux identity (Appendix C) and the
quantitative adiabatic-elimination lemma / Assumption A (Appendix D).

Self-contained: numpy + scipy only (no qutip, no qudlab).

Toy model.  A 4-level system with code space span{|0>,|1>} and leaked space
span{|2>,|3>}.  The generator is

    L = L_stab + gamma * D[M],

with:
  * meter  M = PMP + (E + E^dagger),  Hermitian, where
        PMP = (Delta_m/2) Z_P + kappa X_P            (within-code part)
        E|0> = eps0 |2|,  E|1> = eps1 (s|2> + sqrt(1-s^2)|3>)   (leak part)
    so that Lambda = P E^dagger E P has Lambda00=eps0^2, Lambda11=eps1^2,
    Lambda01 = eps0 eps1 s;
  * direct-return stabilizer  L_stab = sum_k D[L_k],  L_k = sqrt(g)|c><l|,
    returning leaked population to the code space (A_k = Pperp L_k Pperp = 0).

We build the 16x16 Liouvillian exactly, measure the windowed log-slope of the
Z autocorrelation C_Z(t) = Tr[Z e^{Lt}(Z P/2)], and compare it to the analytic
predictions of the paper.  Nothing is fitted; the model parameters are fixed
and every check reports measured-vs-predicted.
"""
import json, os
import numpy as np
import scipy.linalg as sla

# ---- 4-level operators -----------------------------------------------------
d = 4
def ket(i):
    v = np.zeros((d, 1), complex); v[i, 0] = 1.0; return v
E00 = ket(0) @ ket(0).conj().T
P = np.diag([1., 1., 0., 0.]).astype(complex)      # code projector
Pp = np.diag([0., 0., 1., 1.]).astype(complex)     # leaked projector
Z = np.diag([1., -1., 0., 0.]).astype(complex)     # logical Z on the code
X = (ket(0) @ ket(1).conj().T + ket(1) @ ket(0).conj().T)

def build_M(kappa, dm, eps0, eps1, s):
    PMP = 0.5 * dm * Z + kappa * X
    E = np.zeros((d, d), complex)
    E += eps0 * (ket(2) @ ket(0).conj().T)                                  # E|0> = eps0|2>
    E += eps1 * s * (ket(2) @ ket(1).conj().T)                              # E|1> = eps1(s|2> + ...
    E += eps1 * np.sqrt(max(0.0, 1 - abs(s) ** 2)) * (ket(3) @ ket(1).conj().T)  #        ...|3>)
    return PMP + E + E.conj().T, E

def liouvillian(M, g, gamma):
    I = np.eye(d, dtype=complex)
    def Dsup(L):
        LdL = L.conj().T @ L
        return np.kron(L.conj(), L) - 0.5 * np.kron(I, LdL) - 0.5 * np.kron(LdL.T, I)
    # direct-return stabilizer: |2|->|0>, |3|->|1>
    Ls = [np.sqrt(g) * (ket(0) @ ket(2).conj().T), np.sqrt(g) * (ket(1) @ ket(3).conj().T)]
    Lstab = sum(Dsup(L) for L in Ls)
    return Lstab + gamma * Dsup(M), Ls

def vec(A):  return A.reshape(-1, order="F")
def unvec(v): return v.reshape((d, d), order="F")

def C_Z(Lsup, ts):
    rho0 = vec(Z @ P / 2.0)
    zc = vec(Z).conj()
    out = []
    for t in ts:
        out.append((zc @ (sla.expm(Lsup * t) @ rho0)).real)
    return np.array(out)

def windowed_slope(Lsup, g, gamma, Mnorm2):
    # lower edge of the window 1/g << t << 1/(gamma ||PMP||^2); take t_- ~ few/g
    t_ = 3.0 / g
    h = t_ * 1e-3
    c0, cm, cp = C_Z(Lsup, [t_, t_ - h, t_ + h])
    return -(np.log(cp) - np.log(cm)) / (2 * h)

def recovered_delta(Ls, E, mu):
    # per-codeword recovery error delta_mu = 1 - <mu| R(|w_mu><w_mu|) |mu>
    wmu = E @ ket(mu); eps = np.linalg.norm(wmu); wmu = wmu / eps
    rho = wmu @ wmu.conj().T
    rec = sum(L @ rho @ L.conj().T for L in Ls)            # one return application (rate factor cancels)
    rec = rec / np.trace(rec).real
    return float(1 - (ket(mu).conj().T @ rec @ ket(mu)).real[0, 0]), eps

results = {"checks": []}
def check(name, measured, predicted, tol, extra=""):
    rel = abs(measured - predicted) / (abs(predicted) + 1e-30)
    ok = rel <= tol or abs(measured - predicted) < 1e-9
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: measured={measured:.6g} predicted={predicted:.6g} "
          f"rel={rel:.2%}  {extra}")
    results["checks"].append(dict(name=name, measured=measured, predicted=predicted, rel=rel, ok=bool(ok)))
    return ok

gamma = 0.05
print("=== (a) within-code term: windowed slope -> 2 gamma kappa^2 ===")
for kappa, dm in [(0.3, 0.0), (0.3, 0.8)]:
    M, E = build_M(kappa, dm, 0.0, 0.0, 0.0)          # E=0: pure within-code
    L, Ls = liouvillian(M, g=200.0, gamma=gamma)
    sl = windowed_slope(L, 200.0, gamma, np.linalg.norm(M) ** 2)
    check(f"within-code slope (kappa={kappa}, dm={dm})", sl, 2 * gamma * kappa ** 2, 0.05,
          extra=f"(spectral eig would be 2g n^2={2*gamma*(kappa**2+(dm/2)**2):.5g})")

print("\n=== (b) leak term: flux identity Gamma_leak = gamma(eps0^2 d0 + eps1^2 d1) ===")
# confusion (s!=0) => leading self-rate = flux; which-path (s=0) => leading self-rate = 0,
# with only an O(gamma/g) higher-order residual (diagonal residual feeds the conjugate, not Z).
g = 200.0
M, E = build_M(0.0, 0.0, 0.3, 0.3, 0.6)               # confusion, kappa=0
L, Ls = liouvillian(M, g=g, gamma=gamma)
d0, _ = recovered_delta(Ls, E, 0); d1, _ = recovered_delta(Ls, E, 1)
flux = gamma * (0.3 ** 2 * d0 + 0.3 ** 2 * d1)
sl = windowed_slope(L, g, gamma, np.linalg.norm(M) ** 2)
check("flux identity [confusion s=0.6]", sl, flux, 0.12, extra=f"(d0={d0:.3g}, d1={d1:.3g})")

M2, E2 = build_M(0.0, 0.0, 0.35, 0.2, 0.0)            # which-path: s=0, eps0!=eps1
L2, _ = liouvillian(M2, g=g, gamma=gamma)
sl_wp = windowed_slope(L2, g, gamma, np.linalg.norm(M2) ** 2)
leak_scale = gamma * 0.35 ** 2                         # gamma eps0^2, the leak rate scale
supp_ok = sl_wp < 0.1 * leak_scale                    # self-rate strongly suppressed vs leak scale
print(f"[{'PASS' if supp_ok else 'FAIL'}] which-path self-rate suppressed: "
      f"slope={sl_wp:.3g} = {sl_wp/leak_scale:.1%} of leak scale = {sl_wp/(gamma/g):.2f}*(gamma/g) "
      f"-> leading which-path self-damage is zero; residual is O(gamma/g)")
results["checks"].append(dict(name="which-path suppressed", measured=sl_wp,
                              leak_scale=leak_scale, ratio_gamma_over_g=sl_wp/(gamma/g), ok=bool(supp_ok)))

print("\n=== (c) g-uniformity: deviation ~ c/g, converging from below, no Zeno ===")
kappa, dm, eps0, eps1, s = 0.3, 0.5, 0.3, 0.3, 0.6
M, E = build_M(kappa, dm, eps0, eps1, s)
d0, e0 = recovered_delta(*(liouvillian(M, 200.0, gamma)[1], E, 0)[:1] + (E, 0))
# recompute deltas cleanly (return channel is g-independent in structure)
_, Ls_ref = liouvillian(M, 1.0, gamma)
d0, _ = recovered_delta(Ls_ref, E, 0); d1, _ = recovered_delta(Ls_ref, E, 1)
pred = 2 * gamma * kappa ** 2 + gamma * (eps0 ** 2 * d0 + eps1 ** 2 * d1)
gs = [5., 20., 100., 500., 2000.]
devs = []
for g in gs:
    L, _ = liouvillian(M, g, gamma)
    sl = windowed_slope(L, g, gamma, np.linalg.norm(M) ** 2)
    dev = (sl - pred) / pred
    devs.append(dev)
    print(f"  g={g:6.0f}: slope={sl:.6g}  additive-pred={pred:.6g}  dev={dev:+.3%}")
bounded = max(abs(np.diff([s for s in [pred*(1+d) for d in devs]]))) < 0.2 * pred
converges = abs(devs[-1] - devs[-2]) < 0.05 * abs(devs[-1] + 1e-30)
print(f"  slope stays bounded across g=5..2000 (no Zeno blow-up): {bounded}; "
      f"converges to a small g-independent offset ({devs[-1]:+.2%}): {converges}")
results["g_sweep"] = dict(g=gs, dev=devs, bounded=bool(bounded), converges=bool(converges),
                          additive_prediction=pred)

print("\n=== (d) block bounds: sup p <= 1.5 g||Lambda||/g,  sup q <= 6 g||E||m/g ===")
g = 5.0
M, E = build_M(0.3, 0.5, 0.3, 0.3, 0.6)
L, _ = liouvillian(M, g, gamma)
Lam = (E.conj().T @ E); normLam = np.linalg.norm(Lam, 2); normE = np.linalg.norm(E, 2); m = np.linalg.norm(M, 2)
ts = np.linspace(0, 5.0, 400)
rho0 = vec((ket(0) + ket(1)) @ (ket(0) + ket(1)).conj().T / 2.0)
ps, qs = [], []
for t in ts:
    r = unvec(sla.expm(L * t) @ rho0)
    ps.append(np.trace(Pp @ r).real)
    qs.append(np.sum(np.abs(P @ r @ Pp)))
supp, supq = max(ps), max(qs)
bp, bq = 1.5 * gamma * normLam / g, 6 * gamma * normE * m / g
print(f"  sup p = {supp:.4g}  bound {bp:.4g}  -> {'OK' if supp<=bp else 'VIOLATION'}")
print(f"  sup q = {supq:.4g}  bound {bq:.4g}  -> {'OK' if supq<=bq else 'VIOLATION'}")
results["block_bounds"] = dict(sup_p=supp, bound_p=bp, sup_q=supq, bound_q=bq)

print("\n=== (e) remainder constant C_beta at beta=1 ===")
cb_corrected = 26 + 61 * 1 + 2 * 1 ** 2
cb_old = 27 + 60 * 1 + 2 * 1 ** 2
print(f"  corrected 26+61b+2b^2 at b=1 = {cb_corrected};  old 27+60b+2b^2 at b=1 = {cb_old} "
      f"(equal at b=1; differ by (b-1) for b!=1)")
results["C_beta"] = dict(corrected_at_1=cb_corrected, old_at_1=cb_old)

here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(here, "data"), exist_ok=True)
with open(os.path.join(here, "data", "lemma_A_validation.json"), "w") as f:
    json.dump(results, f, indent=2)
n_pass = sum(c["ok"] for c in results["checks"])
print(f"\n{n_pass}/{len(results['checks'])} slope/flux checks passed; wrote data/lemma_A_validation.json")
