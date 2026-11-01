"""Figure 1 of the MaCVi Diver3D team report. Values transcribed from logged analyses (EXPERIMENTS.md rounds
1-16 and the analysis logs): leaderboard val/test pairs, mixed-effects recording effects on log box width
(val, ensemble mixed7, covariates range / log range / along-beam and vertical components of the body axis),
and out-of-fold found@BEV0.35 vs per-pass NMS threshold (fold detectors on 14 held-out train recordings,
mirror fusion + track smoothing)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"           # validated categorical slots 1-3 (light surface)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.5,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "axes.labelcolor": INK, "text.color": INK, "legend.fontsize": 6.8, "legend.frameon": False})

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]})
for a in ax:
    a.grid(True, color=GRID, linewidth=0.5, linestyle="-"); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)

# (a) validation vs test SDS of our uploads
ens = {"ens3-ref": (0.7405, 0.7235), "ens7-ref": (0.7550, 0.7264), "ens7-s2": (0.7581, 0.7246), "ens9-occl": (0.7541, 0.7277),
       "ens13-occl": (0.7543, 0.7229), "occl8": (0.7448, 0.7145), "fp6": (0.7485, 0.7217), "a45x7": (0.7508, 0.7137), "ens9-sc105": (0.7599, 0.7290), "last6": (0.7361, 0.6973)}
single = {"single-ref": (0.7320, 0.6954)}
lo, hi = 0.69, 0.765
a = ax[0]
a.plot([lo, hi], [lo, hi], color=INK2, linewidth=0.6)
a.text(0.7262, 0.7345, "test = val", color=INK2, fontsize=6.5, ha="left", va="bottom")
v = np.array([p[0] for p in ens.values()]); t = np.array([p[1] for p in ens.values()])
a.scatter(v, t, s=22, color=S1, edgecolor=SURF, linewidth=1.0, zorder=3, label="ensembles")
a.scatter([0.7320], [0.6954], s=22, color=S2, edgecolor=SURF, linewidth=1.0, zorder=3, label="single model")
a.annotate("best: 0.7290", (0.7599, 0.7290), xytext=(0.7465, 0.737), fontsize=6.5, color=INK,
           arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.5))
a.set_xlim(0.725, 0.765); a.set_ylim(lo, 0.755)
a.set_xlabel("validation SDS"); a.set_ylabel("hidden-test SDS")
a.set_title("(a) Validation vs. test, our uploads", loc="left")
a.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2)

# (b) recording effects on box width: labels vs our predictions
rec = ["0000", "0043", "0048", "0068", "0075", "0080", "0082"]
gt_w = np.array([0.340, 0.237, -0.088, 0.135, -0.172, -0.541, 0.089])
pr_w = np.array([-0.042, 0.098, -0.045, -0.008, 0.026, 0.002, -0.031])
a = ax[1]
x = np.arange(len(rec)); w = 0.36
a.bar(x - w / 2 - 0.01, 100 * (np.exp(gt_w) - 1), width=w, color=S1, edgecolor=SURF, linewidth=0.6, label="labelled width", zorder=3)
a.bar(x + w / 2 + 0.01, 100 * (np.exp(pr_w) - 1), width=w, color=S2, edgecolor=SURF, linewidth=0.6, label="our predicted width", zorder=3)
a.axhline(0, color=INK2, linewidth=0.6, zorder=2)
a.set_xticks(x); a.set_xticklabels(rec); a.set_xlabel("validation recording")
a.set_ylabel("recording effect on width (%)")
a.set_ylim(-50, 48)
a.set_title("(b) Width offsets by recording", loc="left")
a.legend(loc="lower left", handletextpad=0.3, borderaxespad=0.2)

# (c) close-pair recovery vs per-pass NMS threshold (out-of-fold)
thr = [0.1, 0.2, 0.3, 0.5]
close = [0.638, 0.795, 0.830, 0.832]; mid = [0.785, 0.811, 0.813, 0.815]; iso = [0.864, 0.865, 0.866, 0.873]
a = ax[2]
for ys, c, lab in ((close, S1, "neighbour < 1 m"), (mid, S2, "1 to 1.5 m"), (iso, S3, "isolated")):
    a.plot(thr, ys, color=c, linewidth=1.4, solid_capstyle="round", zorder=3, label=lab)
    a.scatter(thr, ys, s=16, color=c, edgecolor=SURF, linewidth=0.9, zorder=4)
a.text(0.505, 0.877, "isolated", color=INK, fontsize=6.5, va="bottom", ha="right")
a.axvline(0.1, color=INK2, linewidth=0.5, zorder=1); a.text(0.107, 0.897, "our setting", color=INK2, fontsize=6.3, va="top")
a.legend(loc="lower right", handletextpad=0.3, borderaxespad=0.2, handlelength=1.4)
a.set_xlim(0.07, 0.53); a.set_ylim(0.62, 0.9); a.set_xticks(thr)
a.set_xlabel("per-pass BEV NMS IoU threshold"); a.set_ylabel("divers found (BEV IoU 0.35)")
a.set_title("(c) Close pairs, held-out recordings", loc="left")

fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig("fig/overview.pdf"); fig.savefig("fig/overview.png", dpi=220)
print("saved")
