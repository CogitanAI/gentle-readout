"""Figure 1 for the readability paper (editor's 3-panel spec).

(a) concept: meter decomposition and the two damage channels
(b) atlas: (protection, readability) plane + GKP lattice curve + number cloud
(c) bound verification: measured self-rate vs the theorem, the d=2 cells

Data = the paper's own validated table values. Output: readability_fig1.pdf

REVISION 2026-08-21
  * Panel (c) self-rates synchronised with Table II after the extractor
    normalization fix: GKP-n 9.5e-2 -> 1.1e-1 and GKP-p 4.7e-2 -> 5.0e-2. The old
    values were produced by the pre-fix windowed estimator, which divided by C(0)
    instead of C(t) and so ran low by exp(-Gamma t) on the fast cells. The figure
    previously disagreed with the table it illustrates.
  * Panel (c) legend now defines the markers (they encode cat vs GKP and were
    undocumented), and the two saturating cells are called out, since "saturation
    occurs where the within-code term binds" is the panel's actual message.
  * Panel (b) legend moved out of the number-code cloud it was covering; the two
    curve annotations repositioned off the curve.
  * Panel (a) laid out on an explicit grid so no label overflows the code-space
    box, which it previously did.
  * Serif mathtext throughout to sit with the REVTeX body text.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

BLUE, RED, GREEN, PURPLE = "#1a6faf", "#b02a2a", "#3a7d44", "#7a52a3"

fig = plt.figure(figsize=(7.0, 2.75))

# --- (a) concept schematic ---------------------------------------------------
ax = fig.add_subplot(131)
ax.axis("off")
ax.set_title("(a) meter vs code", fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# code-space box, sized to the widest label it has to hold
ax.add_patch(plt.Rectangle((0.00, 0.40), 0.43, 0.48, fc="#dce9f7",
                           ec="k", lw=0.8))
ax.text(0.215, 0.795, "code space $P$", ha="center", va="center", fontsize=6.6)
ax.text(0.215, 0.665, r"$\Delta m$: signal", ha="center", va="center",
        fontsize=6.0, color=BLUE)
ax.text(0.215, 0.545, r"$\kappa$: rotation", ha="center", va="center",
        fontsize=6.0, color=RED)
ax.text(0.215, 0.455, r"$\Rightarrow 2\gamma\kappa^{2}$", ha="center",
        va="center", fontsize=6.0, color=RED)

# leak out of the code space, then forking into its two fates -- the fork is
# the point of the panel, so it is drawn rather than implied by adjacency
FORK = (0.60, 0.64)
ax.annotate("", xy=FORK, xytext=(0.44, 0.64),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
# the gap between the box and the fork is narrow and both fork arrows pass
# close to it, so the label carries an opaque background rather than relying on
# clearance that changes whenever the layout is retuned
ax.text(0.515, 0.658, r"$E$ leak", ha="center", va="bottom", fontsize=6.4,
        color=GREEN, zorder=5,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.9))
ax.annotate("", xy=(0.71, 0.87), xytext=FORK,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
ax.annotate("", xy=(0.71, 0.33), xytext=FORK,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))

ax.text(0.74, 0.87, "KL-correctable:\nrepaired free", ha="left", va="center",
        fontsize=6.0, color=GREEN, linespacing=1.4)
ax.text(0.74, 0.30,
        r"confusion $|\Lambda_{01}|$" + "\n" +
        r"$\Rightarrow \gamma\,\varepsilon_{\rm KL}^{2}/2$",
        ha="left", va="center", fontsize=6.0, color=RED, linespacing=1.4)

# --- (b) the atlas -----------------------------------------------------------
ax = fig.add_subplot(132)
ax.set_title("(b) readability vs protection", fontsize=9)

# GKP lattice aspect sweep (spectral extraction)
lat_prot = [1.36, 1.50, 1.67, 1.50, 1.35]
lat_perc = [789, 277, 92.6, 40.1, 24.0]
ax.plot(lat_prot, lat_perc, "-o", ms=3.2, lw=1.1, color=PURPLE,
        label="GKP lattice $r$", zorder=3)
ANN = dict(fontsize=6.3, color=PURPLE, va="center",
           arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.6,
                           shrinkA=1, shrinkB=2))
ax.annotate("square", xy=(1.67, 92.6), xytext=(1.73, 26), ha="left", **ANN)
ax.annotate("$r{=}0.75$", xy=(1.36, 789), xytext=(0.72, 2100), ha="left", **ANN)

ax.scatter([1.12], [650], marker="*", s=70, color=BLUE, zorder=4,
           label="cat $d{=}2$ (via $q$)")
ax.scatter([1.43], [83], marker="s", s=26, color=GREEN, zorder=4,
           label="GKP $d{=}2$ (via mod-$q$)")

# INTEGRITY NOTE (2026-07-19 review): the number-code cloud is the REAL 48-code
# search output from scratch/round10_stageB.py (3 g-values x 4 M-values x 4
# envelopes), rerun live under qutip 5.3.0; it previously used np.random points.
# CAVEATS (stated in the caption): protection uses the disclosed biased/
# conservative recovery pump (LAM=1.0); P is the spectral-extractor self-rate,
# not the autocorrelation path used for the d=2 anchors. Three codes returned
# P ~ 0 and are clipped to the axis floor 0.05.
np_prot = [1.27, 1.27, 1.27, 1.27, 1.02, 1.02, 1.03, 1.01, 0.86, 0.85,
           0.89, 0.85, 0.74, 0.72, 0.79, 0.73, 1.00, 1.00, 1.00, 1.00,
           0.80, 0.77, 0.81, 0.78, 0.65, 0.60, 0.69, 0.63, 0.54, 0.48,
           0.60, 0.52, 0.80, 0.80, 0.80, 0.80, 0.60, 0.57, 0.62, 0.58,
           0.47, 0.41, 0.51, 0.43, 0.36, 0.34, 0.42, 0.33]
np_perc = [1.1, 1.1, 1.1, 1.1, 0.6, 0.4, 0.7, 0.5, 0.4, 0.3,
           0.6, 0.3, 0.4, 0.2, 0.6, 0.3, 1.2, 1.2, 1.2, 1.2,
           0.8, 0.6, 1.0, 0.7, 0.7, 0.5, 0.9, 0.5, 0.6, 0.05,
           0.9, 0.5, 1.6, 1.6, 1.6, 1.6, 1.3, 1.0, 1.5, 1.0,
           1.0, 0.1, 1.3, 0.1, 0.9, 0.05, 1.5, 0.05]
ax.scatter(np_prot, np_perc, marker=".", s=12, color="0.6", zorder=2,
           label="number codes (48)")

ax.set_yscale("log")
ax.set_xlim(0.25, 2.10)   # headroom so the "square" label is not clipped
ax.set_ylim(6e-3, 6e3)   # floor dropped to clear the legend off the cloud:
                         # four number codes sit at (1.27, 1.1), which the
                         # lower-right legend otherwise hides
ax.set_xlabel(r"protection $-\log_{10}\Gamma_{\rm worst}$", fontsize=8)
ax.set_ylabel(r"readability $\mathcal{P}$", fontsize=8)
ax.tick_params(labelsize=7)
# lower right is the empty corner: the number-code cloud stops at x ~ 1.3 and
# the lattice curve stays above P ~ 20, so nothing lives there. Upper left is
# NOT free -- it holds the r = 0.75 callout and the top of the lattice curve.
ax.legend(fontsize=5.4, loc="lower right", framealpha=0.92, borderpad=0.4,
          handletextpad=0.5, labelspacing=0.35)

# --- (c) bound verification --------------------------------------------------
ax = fig.add_subplot(133)
ax.set_title(r"(c) Theorem 1, all $d{=}2$ cells", fontsize=9)

# (label, measured Gamma_self, 2*gamma*kappa^2, gamma*eps_KL^2/2) -- Table II.
CELLS = [
    ("cat",  "n",      4.8e-3, 7.2e-4, 3.8e-4),
    ("cat",  "q",      1.2e-3, 1.9e-4, 1.8e-4),
    ("cat",  "p",      1.5e-3, 1.9e-4, 1.8e-4),
    ("cat",  "parity", 1.0e-1, 1.0e-1, 0.0),
    ("cat",  "mod-q",  1.7e-3, 4.0e-5, 1.2e-5),
    ("cat",  "mod-p",  9.4e-3, 2.3e-3, 4.8e-4),
    ("GKP",  "n",      1.1e-1, 1.6e-2, 5.4e-3),
    ("GKP",  "q",      3.1e-3, 0.0,    9.8e-6),
    ("GKP",  "p",      5.0e-2, 0.0,    3.9e-3),
    ("GKP",  "parity", 1.4e-8, 0.0,    0.0),
    ("GKP",  "mod-q",  1.7e-3, 2.0e-4, 2.1e-4),
    ("GKP",  "mod-p",  7.4e-2, 7.1e-2, 3.4e-7),
]

lo, hi = 1e-5, 0.4
ax.fill_between([lo, hi], [lo, hi], lo, color="0.9", zorder=0)
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, zorder=1, label="equality")

for code, meter, self_m, bk, bl in CELLS:
    bound = max(bk, bl)
    if bound <= 1e-12:
        continue                       # blind cell: bound is exactly zero
    marker = "o" if code == "cat" else "s"
    ax.plot(bound, self_m, marker, ms=4.2, mfc="none", mec=RED, mew=1.1,
            zorder=3)

# legend proxies: the markers carry information and were previously undefined
ax.plot([], [], "o", ms=4.2, mfc="none", mec=RED, mew=1.1, label="cat")
ax.plot([], [], "s", ms=4.2, mfc="none", mec=RED, mew=1.1, label="GKP")

ax.text(3e-4, 2.2e-5, "forbidden", fontsize=7, color="0.4")
ax.annotate("saturation", xy=(9.0e-2, 1.05e-1), xytext=(4.5e-3, 2.4e-1),
            fontsize=6.3, color="0.25", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.6,
                            shrinkA=1, shrinkB=3))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel(r"$\gamma\max(2\kappa^{2},\ \varepsilon_{\rm KL}^{2}/2)$",
              fontsize=8)
ax.set_ylabel(r"measured $\Gamma_{\rm self}$", fontsize=8)
ax.tick_params(labelsize=7)
ax.legend(fontsize=6, loc="lower right", framealpha=0.92, borderpad=0.4,
          handletextpad=0.5, labelspacing=0.35)

fig.tight_layout(pad=0.6)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "readability_fig1.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", out, flush=True)
