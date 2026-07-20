"""Figure 1 for the readability paper (editor's 3-panel spec).
(a) concept: meter decomposition and the two damage channels
(b) atlas: (protection, readability) plane + GKP lattice curve + number cloud
(c) bound verification: measured self vs the theorem, 12 d=2 cells
Data = the paper's own validated table values. Output: readability_fig1.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(7.0, 2.6))

# --- (a) concept schematic --------------------------------------------------
ax = fig.add_subplot(131)
ax.axis("off")
ax.set_title("(a) meter vs code", fontsize=9)
ax.add_patch(plt.Rectangle((0.05, 0.35), 0.42, 0.38, fc="#dce9f7",
                           ec="k", lw=0.8))
ax.text(0.26, 0.66, "code space $P$", ha="center", fontsize=7)
ax.text(0.26, 0.52, r"$\Delta m$: signal", ha="center", fontsize=7,
        color="#1a6faf")
ax.text(0.26, 0.42, r"$\kappa$: rotation $\Rightarrow 2\gamma\kappa^2$",
        ha="center", fontsize=7, color="#b02a2a")
ax.annotate("", xy=(0.88, 0.72), xytext=(0.47, 0.60),
            arrowprops=dict(arrowstyle="->", color="#3a7d44", lw=1.2))
ax.text(0.72, 0.74, r"$E$ leak", fontsize=7, color="#3a7d44")
ax.text(0.66, 0.47, "KL-correctable:\nrepaired free", fontsize=6.5,
        ha="center", color="#3a7d44")
ax.text(0.66, 0.18,
        r"confusion $|\Lambda_{01}|$" + "\n" +
        r"$\Rightarrow \gamma\,\varepsilon_{\rm KL}^2/2$",
        fontsize=6.5, ha="center", color="#b02a2a")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# --- (b) the atlas -----------------------------------------------------------
ax = fig.add_subplot(132)
ax.set_title("(b) readability vs protection", fontsize=9)
# GKP lattice aspect sweep (spectral extraction)
lat_prot = [1.36, 1.50, 1.67, 1.50, 1.35]
lat_perc = [789, 277, 92.6, 40.1, 24.0]
ax.plot(lat_prot, lat_perc, "-o", ms=3.5, color="#7a52a3",
        label="GKP lattice $r$", zorder=3)
ax.annotate("square", xy=(1.67, 92.6), fontsize=6.5,
            xytext=(1.55, 160), color="#7a52a3")
ax.annotate("$r{=}0.75$", xy=(1.36, 789), fontsize=6.5,
            xytext=(1.15, 900), color="#7a52a3")
ax.scatter([1.12], [650], marker="*", s=70, color="#1a6faf", zorder=4,
           label="cat $d{=}2$ (via $q$)")  # was 700; corrected to Table I value 650
ax.scatter([1.43], [83], marker="s", s=30, color="#3a7d44", zorder=4,
           label="GKP $d{=}2$ (via mod-$q$)")
# INTEGRITY FIX (2026-07-19 review): the number-code cloud now uses the REAL 48-code search
# outputs from round10_stageB.py (3 g-values x 4 M-values x 4 envelopes = 48 codes),
# reran live under qutip 5.3.0. It previously used np.random points (kept below, commented, so
# nothing is lost). x = protection = -log10(Gamma_worst); y = readability P.
# CAVEATS (state in caption/text): protection uses the disclosed biased/conservative recovery
# pump (LAM=1.0); P is the spectral-extractor self-rate (not the autocorr path used for the d=2
# anchors). Three codes returned P~0 (rounded); shown clipped to the axis floor 0.05.
# --- OLD fabricated cloud (do not use; retained for provenance) ---
# rng = np.random.default_rng(7)
# np_prot = rng.uniform(0.33, 1.27, 20)
# np_perc = 10 ** rng.uniform(-0.7, 0.2, 20)
# --- REAL 48-code data ---
np_prot = [1.27, 1.27, 1.27, 1.27, 1.02, 1.02, 1.03, 1.01, 0.86, 0.85,
           0.89, 0.85, 0.74, 0.72, 0.79, 0.73, 1.00, 1.00, 1.00, 1.00,
           0.80, 0.77, 0.81, 0.78, 0.65, 0.60, 0.69, 0.63, 0.54, 0.48,
           0.60, 0.52, 0.80, 0.80, 0.80, 0.80, 0.60, 0.57, 0.62, 0.58,
           0.47, 0.41, 0.51, 0.43, 0.36, 0.34, 0.42, 0.33]
np_perc = [1.1, 1.1, 1.1, 1.1, 0.6, 0.4, 0.7, 0.5, 0.4, 0.3,
           0.6, 0.3, 0.4, 0.2, 0.6, 0.3, 1.2, 1.2, 1.2, 1.2,
           0.8, 0.6, 1.0, 0.7, 0.7, 0.5, 0.9, 0.5, 0.6, 0.05,
           0.9, 0.5, 1.6, 1.6, 1.6, 1.6, 1.3, 1.0, 1.5, 1.0,
           1.0, 0.1, 1.3, 0.1, 0.9, 0.05, 1.5, 0.05]  # three 0.0 -> 0.05 (axis floor)
ax.scatter(np_prot, np_perc, marker=".", s=12, color="0.6", zorder=2,
           label="number codes (48)")
ax.set_yscale("log")
ax.set_xlabel(r"protection $-\log_{10}\Gamma_{\rm worst}$", fontsize=8)
ax.set_ylabel(r"readability $\mathcal{P}$", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=5.5, loc="lower left", framealpha=0.9)

# --- (c) bound verification ---------------------------------------------------
ax = fig.add_subplot(133)
ax.set_title("(c) Theorem 1, all $d{=}2$ cells", fontsize=9)
self_m = [4.8e-3, 1.2e-3, 1.5e-3, 1.0e-1, 1.7e-3, 9.4e-3,
          9.5e-2, 3.1e-3, 4.7e-2, 1.4e-8, 1.7e-3, 7.4e-2]  # last: was 7.0e-2; Table II GKP mod-p = 7.4e-2
bound = [max(a, b) for a, b in
         [(7.2e-4, 3.8e-4), (1.9e-4, 1.8e-4), (1.9e-4, 1.8e-4),
          (1.0e-1, 0), (4.0e-5, 1.2e-5), (2.3e-3, 4.8e-4),
          (1.6e-2, 5.4e-3), (1e-30, 9.8e-6), (1e-30, 3.9e-3),
          (1e-30, 1e-30), (2.0e-4, 2.1e-4), (7.1e-2, 3.4e-7)]]
mk = ["o"] * 6 + ["s"] * 6
for x, y, m in zip(bound, self_m, mk):
    if x > 1e-12:
        ax.plot(x, y, m, ms=4, mfc="none", mec="#b02a2a", mew=1.1)
lo, hi = 1e-5, 0.3
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="equality")
ax.fill_between([lo, hi], [lo, hi], lo, color="0.9", zorder=0)
ax.text(2e-4, 3e-5, "forbidden", fontsize=7, color="0.4")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel(r"$\gamma\max(2\kappa^2,\ \varepsilon_{\rm KL}^2/2)$",
              fontsize=8)
ax.set_ylabel(r"measured $\Gamma_{\rm self}$", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6, loc="upper left")

fig.tight_layout(pad=0.6)
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "readability_fig1.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out, flush=True)
