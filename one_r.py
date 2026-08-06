"""Run a SINGLE aspect ratio of round10_stageA in a fresh process.

Does not modify the released script: imports it and calls score() unchanged,
with the same N, D and module-level constants. Purpose is only to give the
expensive r>1 points a clean heap, since the in-sequence run died in SuperLU's
allocator on the fourth factorization.

Usage: python one_r.py 1.18
"""
import sys
import os

REL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REL)
os.chdir(REL)

import round10_stageA as A  # noqa: E402

r = float(sys.argv[1])
print(f"single-point run: r={r}  N={A.N} D={A.D} eps={A.EPS}", flush=True)
try:
    out, nb, k0, k1, res = A.score(A.N, A.D, r)
except Exception as e:
    print(f"{r:6.2f}  FAILED: {type(e).__name__}: {e}", flush=True)
    sys.exit(2)

if out is None:
    print(f"{r:6.2f} {nb:7.2f} {k0:9.1e} {k1:7.1e}   CARD FAIL (res={res:.1e})",
          flush=True)
    sys.exit(3)

P_prot, P_perc, via, worst = out
print(f"{r:6.2f} {nb:7.2f} {k0:9.1e} {k1:7.1f} {P_prot:7.2f} "
      f"{P_perc:8.2f} {via:>7s} {worst:10.2e}", flush=True)
