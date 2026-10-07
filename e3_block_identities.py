"""Machine-check the finite-dimensional steps of Lemma 2 (main text App. D; full
proof in the Supplementary Material, Secs. S1-S5).
Every identity is checked
on random instances (random dimension, random Hermitian meter, random stabilizer
jumps satisfying the dark-space condition (G1)) to machine precision, and every
norm inequality is checked as an inequality over many draws.

Checks:
  (B2)  D^dag[M](P_perp) = Lambda (+) (-E E^dag) (+) cross,  cross = 1/2 E^dag M_perp - 1/2 M_c E^dag
        and ||cross|| <= ||E|| m
  (B3)  D^dag[M](Z_L) = [D^dag[M_c](Z) - 1/2{Lambda, Z}] (+) (E Z_L E^dag) (+) cross'
        cross' = M_c Z_L E^dag - 1/2 Z_L (M_c E^dag + E^dag M_perp),  ||cross'|| <= 2 ||E|| m
  (15)  G = K_perp - sum A^dag A = sum B^dag B   (exact, needs (G1))
  (15') d/dtau tr[e^{L'_perp tau} s] = -tr[G e^{L'_perp tau} s]
  (ii)  R(s) = int J_c(e^{L'_perp tau} s) dtau is CP and EXACTLY trace preserving,
        including for CASCADED stabilizers (A_k != 0), whenever e^{L'_perp tau} -> 0.

Run:  python -u e3_block_identities.py
"""
import numpy as np
import scipy.linalg

rng = np.random.default_rng(20261002)
TRIALS = 200
TOL = 1e-10


def herm(n):
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return (X + X.conj().T) / 2


def Ddag(L, X):
    return L.conj().T @ X @ L - 0.5 * (L.conj().T @ L @ X + X @ L.conj().T @ L)


def blocks(n, dc):
    P = np.zeros((n, n), complex); P[:dc, :dc] = np.eye(dc)
    return P, np.eye(n) - P


def opnorm(X):
    return np.linalg.norm(X, 2)


worst = dict(B2=0.0, B3=0.0, G=0.0, flux=0.0, trR=0.0)
ratio = dict(cross=0.0, crossp=0.0)
for trial in range(TRIALS):
    n = int(rng.integers(4, 9)); dc = 2
    P, Pp = blocks(n, dc)
    M = herm(n)
    m = opnorm(M)
    Mc, E, Mperp = P @ M @ P, Pp @ M @ P, Pp @ M @ Pp
    Lam = E.conj().T @ E
    # (B2)
    lhs = Ddag(M, Pp)
    cross = 0.5 * E.conj().T @ Mperp - 0.5 * Mc @ E.conj().T
    rhs = Lam - E @ E.conj().T + cross + cross.conj().T
    worst["B2"] = max(worst["B2"], np.abs(lhs - rhs).max())
    ratio["cross"] = max(ratio["cross"], opnorm(cross) / (opnorm(E) * m))
    # (B3), Z_L = logical Z on the code block
    Z = np.zeros((n, n), complex); Z[0, 0], Z[1, 1] = 1, -1
    lhs = Ddag(M, Z)
    pp = Ddag(Mc, Z) - 0.5 * (Lam @ Z + Z @ Lam)
    crossp = Mc @ Z @ E.conj().T - 0.5 * Z @ (Mc @ E.conj().T + E.conj().T @ Mperp)
    rhs = pp + E @ Z @ E.conj().T + crossp + crossp.conj().T
    worst["B3"] = max(worst["B3"], np.abs(lhs - rhs).max())
    ratio["crossp"] = max(ratio["crossp"], opnorm(crossp) / (opnorm(E) * m))
    # stabilizer jumps satisfying (G1) L_k P = 0, generically CASCADED (A_k != 0)
    nk = int(rng.integers(1, 4))
    Ls = []
    for _ in range(nk):
        X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        Ls.append(X @ Pp)
    K = sum(L.conj().T @ L for L in Ls)
    A = [Pp @ L @ Pp for L in Ls]
    B = [P @ L @ Pp for L in Ls]
    Kp = Pp @ K @ Pp
    G1 = Kp - sum(a.conj().T @ a for a in A)
    G2 = sum(b.conj().T @ b for b in B)
    worst["G"] = max(worst["G"], np.abs(G1 - G2).max())
    # perp-block superoperator L'_perp and the trace-flux identity
    Vp = np.eye(n)[:, dc:]
    Ap = [Vp.T @ a @ Vp for a in A]; Bp = [P[:dc] @ b @ Vp for b in B]
    Kpp = Vp.T @ Kp @ Vp; Gp = Vp.T @ G2 @ Vp
    k = n - dc; I = np.eye(k)
    Lsup = -0.5 * (np.kron(I, Kpp) + np.kron(Kpp.T, I)) + sum(np.kron(a.conj(), a) for a in Ap)
    s = herm(k); s = s @ s  # positive leaked state
    s /= np.trace(s).real
    for tau in (0.0, 0.3, 1.7):
        st = (scipy.linalg.expm(Lsup * tau) @ s.reshape(-1, order="F")).reshape(k, k, order="F")
        dtr = np.trace((Lsup @ st.reshape(-1, order="F")).reshape(k, k, order="F"))
        worst["flux"] = max(worst["flux"], abs(dtr + np.trace(Gp @ st)))
    # R trace preservation (needs e^{L' tau} -> 0, generic for random jumps)
    if np.max(np.linalg.eigvals(Lsup).real) < -1e-9:
        X = np.linalg.solve(Lsup, -s.reshape(-1, order="F")).reshape(k, k, order="F")
        Rs = sum(b @ X @ b.conj().T for b in Bp)
        worst["trR"] = max(worst["trR"], abs(np.trace(Rs) - 1))
        # CP: R(s) >= 0 for s >= 0
        assert np.linalg.eigvalsh((Rs + Rs.conj().T) / 2).min() > -1e-9, "R not positive"


# ---- Step 1 and Step 2 source bounds, on random states --------------------------
# Step 1: d/dt tr[P_perp rho] from the meter = gamma tr[D^dag[M](P_perp) rho]
#         <= gamma ||Lambda|| + 2 gamma ||E|| m q          (q = ||P rho P_perp||_1)
# Step 2: [D[M] rho]_{c,perp} = (rho_cc-driven part) + rest,
#         rho_cc-driven = M_c rho_cc E^dag - 1/2 rho_cc (M_c E^dag + E^dag M_perp)
#         ||rho_cc-driven||_1 <= 2 ||E|| m tr rho_cc,   ||rest||_1 <= 2 m^2 (2q + p)
# plus the bootstrap closure: for eta <= 1/32 the linear system gives
#         Qbar <= 6 gamma||E||m/g and Pbar <= (3/2) gamma||Lambda||/g.
def tracenorm(X):
    return np.linalg.svd(X, compute_uv=False).sum()


def D(M, r):
    return M @ r @ M - 0.5 * (M @ M @ r + r @ M @ M)


slack = dict(step1=np.inf, drive=np.inf, rest=np.inf)
for trial in range(TRIALS):
    n = int(rng.integers(4, 9)); dc = 2
    P, Pp = blocks(n, dc)
    M = herm(n); m = opnorm(M)
    Mc, E, Mperp = P @ M @ P, Pp @ M @ P, Pp @ M @ Pp
    Lam = E.conj().T @ E
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = X @ X.conj().T
    rho /= np.trace(rho).real
    rcc, rcp, rpp = P @ rho @ P, P @ rho @ Pp, Pp @ rho @ Pp
    q, p = tracenorm(rcp), np.trace(rpp).real
    s1 = np.trace(Ddag(M, Pp) @ rho).real
    slack["step1"] = min(slack["step1"], opnorm(Lam) * np.trace(rcc).real + 2 * opnorm(E) * m * q - s1)
    drive = Mc @ rcc @ E.conj().T - 0.5 * rcc @ (Mc @ E.conj().T + E.conj().T @ Mperp)
    full = P @ D(M, rho) @ Pp
    assert np.abs(P @ D(M, rcc) @ Pp - drive).max() < TOL, "rho_cc-driven block wrong"
    rest = full - drive
    slack["drive"] = min(slack["drive"], 2 * opnorm(E) * m * np.trace(rcc).real - tracenorm(drive))
    slack["rest"] = min(slack["rest"], 2 * m ** 2 * (2 * q + p) - tracenorm(rest))

eta = 1 / 32
# worst case ||E|| = m: Qbar(1 - 8 eta - 8 eta^2) <= 4 (1 + eta) gamma||E||m/g
Qc = 4 * (1 + eta) / (1 - 8 * eta - 8 * eta ** 2)
Pc = 1 + 2 * eta * Qc
print(f"  Step 1 bound slack (min over states)       = {slack['step1']:.3e}  (claim >= 0)")
print(f"  Step 2 rho_cc-driven slack                 = {slack['drive']:.3e}  (claim >= 0)")
print(f"  Step 2 rest slack                          = {slack['rest']:.3e}  (claim >= 0)")
print(f"  bootstrap at eta=1/32: Qbar <= {Qc:.3f} gamma||E||m/g (claim 6), "
      f"Pbar <= {Pc:.3f} gamma||Lambda||/g (claim 1.5)")

ok = (all(v < TOL for v in worst.values()) and min(slack.values()) >= -1e-12
      and Qc <= 6 and Pc <= 1.5)
ok = ok and all(v < TOL for v in worst.values()) and ratio["cross"] <= 1 + 1e-12 and ratio["crossp"] <= 2 + 1e-12
print(f"{TRIALS} random instances (dim 4-8, code dim 2, 1-3 cascaded stabilizer jumps)")
for kk, v in worst.items():
    print(f"  max |lhs - rhs|  {kk:<5s} = {v:.1e}")
print(f"  max ||cross||  / (||E|| m) = {ratio['cross']:.3f}   (claim <= 1)")
print(f"  max ||cross'|| / (||E|| m) = {ratio['crossp']:.3f}   (claim <= 2)")
print("ALL PASS" if ok else "FAILURE")
