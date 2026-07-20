"""Cat-code qudit manifolds: coherent-state legs, code projectors, logical ops.

A d-legged cat code encodes a qudit of dimension d in a single bosonic mode.
The code space is spanned by d coherent states equally spaced on a circle of
radius |alpha| in phase space,

    |alpha_k> = |alpha * omega^k>,   omega = exp(2 pi i / d),   k = 0 .. d-1.

These states are the kernel of the d-photon stabilizing jump operator
L_d = sqrt(kappa_d) (a^d - alpha^d), because a^d |alpha_k> = alpha^d |alpha_k>
(omega^{k d} = 1). They are non-orthogonal (overlap ~ exp(-2|alpha|^2) between
adjacent legs), so a code projector is built by orthonormalizing their span.

The "rotated" logical basis is the discrete Fourier transform of the legs,

    |phi_j> ~ sum_k omega^{-j k} |alpha_k>,

which are the d generalized-parity ("multi-component cat") states. For d = 2
these are the familiar even/odd cats |C_+/->  = (|alpha> +/- |-alpha>)/norm.

References
----------
Mirrahimi et al., New J. Phys. 16, 045014 (2014) — two-photon dissipation.
Grimm et al., Nature 584, 205 (2020) — Kerr-cat.
Albert et al., Phys. Rev. A 94, 032313 (2016) — multi-component (rotation-
    symmetric) bosonic codes.

Units: dimensionless (all rates are relative; time is in units of 1/kappa).
"""

from __future__ import annotations

import numpy as np
import qutip as qt


def coherent_legs(N: int, alpha: complex, d: int) -> list[qt.Qobj]:
    """The d coherent states |alpha * omega^k>, k = 0..d-1, in an N-Fock space."""
    if d < 2:
        raise ValueError("qudit dimension d must be >= 2")
    if N < 2:
        raise ValueError("Fock cutoff N must be >= 2")
    omega = np.exp(2j * np.pi / d)
    return [qt.coherent(N, alpha * omega**k) for k in range(d)]


def _orthonormal_span(vectors: list[qt.Qobj]) -> list[qt.Qobj]:
    """Orthonormal basis (list of kets) for the span of the given kets via QR."""
    N = vectors[0].shape[0]
    mat = np.column_stack([v.full().ravel() for v in vectors])  # N x d
    q, _ = np.linalg.qr(mat)  # columns are orthonormal, span == span(mat)
    return [qt.Qobj(q[:, j].reshape(N, 1)) for j in range(q.shape[1])]


def code_projector(N: int, alpha: complex, d: int) -> qt.Qobj:
    """Projector P onto the d-dimensional cat-code manifold span{|alpha_k>}.

    Built by orthonormalizing the (non-orthogonal) coherent legs, so P is an
    exact rank-d orthogonal projector: P = P^dagger = P^2, Tr(P) = d (up to
    the Fock-truncation error, which is exponentially small when N >> |alpha|^2).
    """
    basis = _orthonormal_span(coherent_legs(N, alpha, d))
    P = sum(e * e.dag() for e in basis)
    return P


def cat_states(N: int, alpha: complex, d: int) -> list[qt.Qobj]:
    """The d rotated logical states |phi_j> (DFT of the coherent legs), normalized.

    For d = 2 these are the even and odd cats. Some components can vanish by
    parity selection when |alpha| is small; any null component is dropped and a
    ValueError is raised only if fewer than d survive (raise N or alpha).
    """
    legs = coherent_legs(N, alpha, d)
    omega = np.exp(2j * np.pi / d)
    out = []
    for j in range(d):
        psi = sum((omega ** (-j * k)) * legs[k] for k in range(d))
        nrm = psi.norm()
        if nrm > 1e-9:
            out.append(psi / nrm)
    if len(out) < d:
        raise ValueError(
            f"only {len(out)}/{d} logical components are non-null at "
            f"alpha={alpha}, d={d}; increase |alpha| or N"
        )
    return out


def logical_paulis_d2(N: int, alpha: float) -> dict[str, qt.Qobj]:
    """Logical X, Y, Z for the d=2 cat qubit, as operators on the full Fock space.

    Uses the orthonormalized {|alpha>, |-alpha>} as computational basis
    |0_L>, |1_L>. Returns operators supported on the code space (zero outside),
    suitable for tracking logical Bloch components via qt.expect against a state.
    """
    e0, e1 = _orthonormal_span(coherent_legs(N, alpha, 2))
    Z = e0 * e0.dag() - e1 * e1.dag()
    X = e0 * e1.dag() + e1 * e0.dag()
    Y = -1j * (e0 * e1.dag() - e1 * e0.dag())
    return {"X": X, "Y": Y, "Z": Z, "I": e0 * e0.dag() + e1 * e1.dag()}


def logical_rotation(N: int, d: int) -> qt.Qobj:
    """Physical d-fold rotation R_d = exp(i 2pi/d * n), the logical shift X_L.

    R_d |alpha omega^k> = |alpha omega^{k+1}>, so it cyclically permutes the
    coherent legs and realizes the qudit shift on the code space exactly and
    natively (it is a passive phase-space rotation). Returned on the full Fock
    space; restrict with the code projector if a strictly-code operator is wanted.
    """
    n = qt.num(N)
    return (1j * (2 * np.pi / d) * n).expm()


def weyl_operators(N: int, alpha: complex, d: int) -> dict[str, qt.Qobj]:
    """Qudit clock (Z) and shift (X) logical operators on the cat-code space.

    Built from the coherent legs and their biorthogonal dual basis {|beta_k>}
    (defined by <beta_k|alpha_l> = delta_kl), so that on the code space

        X_L |alpha_k> = |alpha_{k+1 mod d}>            (shift)
        Z_L |alpha_k> = omega^k |alpha_k>              (clock)

    which satisfy the Weyl-Heisenberg relation Z_L X_L = omega X_L Z_L. Both
    operators are supported on the d-dimensional code space (zero elsewhere).
    For d = 2 these reduce to logical bit-flip (X) and phase-flip (Z). Used to
    label which logical error each slow Liouvillian mode corresponds to.
    """
    legs = coherent_legs(N, alpha, d)
    M = np.column_stack([v.full().ravel() for v in legs])   # N x d
    gram = M.conj().T @ M
    dual = M @ np.linalg.inv(gram)                           # columns |beta_k>
    omega = np.exp(2j * np.pi / d)
    X = np.zeros((N, N), dtype=complex)
    Z = np.zeros((N, N), dtype=complex)
    for k in range(d):
        bra_k = dual[:, k].conj()
        X += np.outer(M[:, (k + 1) % d], bra_k)
        Z += omega**k * np.outer(M[:, k], bra_k)
    return {"X": qt.Qobj(X), "Z": qt.Qobj(Z)}


def auto_cutoff(alpha: float, d: int, margin: float = 4.0) -> int:
    """A safe Fock cutoff N for a cat of amplitude alpha and d legs.

    The largest leg has mean photon number |alpha|^2; the code and its leakage
    dynamics need headroom above that. N = ceil(margin * |alpha|^2) + 8 d + 16
    is comfortably converged for the amplitudes used here (validated against
    N -> 2N invariance of the logical rates in the logical-rate tests).
    """
    return int(np.ceil(margin * abs(alpha) ** 2)) + 8 * d + 16
