"""Generate print-quality PNG figures for the paper (IEEE column width, 300 dpi).

Grayscale-safe by construction: series are separated by lightness AND marker
shape / line weight, never hue alone.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRATCH = Path(__file__).parent
FIGS = Path(__file__).resolve().parents[1] / "new_paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "font.family": "serif",
    "axes.linewidth": 0.6, "axes.edgecolor": "#666666",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "axes.labelcolor": "#222222", "text.color": "#222222",
    "legend.frameon": False,
})

r = json.load(open(SCRATCH / "results_exp4.json"))

# ── Fig: outage failover timeline (dual step plot) ───────────────────────────
LEVEL = {"B": 1, "C": 2, "D": 3}


def rle_to_xy(rle):
    xs, ys, f = [], [], 1
    for mode, cnt in rle:
        xs.append(f)
        ys.append(LEVEL[mode])
        f += cnt
    xs.append(f)          # extend last segment to the end
    ys.append(ys[-1])
    return xs, ys


fig, ax = plt.subplots(figsize=(3.5, 1.75), dpi=300)
o = r["outage"]
a0, a1 = o["outage_at"], o["outage_at"] + o["outage_dur"]
ax.axvspan(a0, a1, color="black", alpha=0.10, lw=0)
ax.text((a0 + a1) / 2, 3.38, "outage", ha="center", va="center",
        fontsize=6.5, style="italic", color="#555555")

xs, ys = rle_to_xy(o["raw_rle"])
ax.step(xs, ys, where="post", color="#999999", lw=0.8, label="raw decision")
xs, ys = rle_to_xy(o["stab_rle"])
ax.step(xs, ys, where="post", color="black", lw=1.7,
        label="stabilized (dwell $k$=5)")

ax.set_xlim(1, 200)
ax.set_ylim(0.55, 3.65)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["B (GPU)", "C (MEC)", "D (cloud)"])
ax.set_xlabel("frame (10 Hz)", fontsize=8)
ax.grid(axis="y", color="#000000", alpha=0.08, lw=0.6)
ax.tick_params(length=2.5, width=0.6, labelsize=7.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower right", fontsize=6.8, handlelength=1.6,
          borderaxespad=0.2, labelspacing=0.25)
fig.tight_layout(pad=0.35)
fig.savefig(FIGS / "outage_timeline.png", dpi=300)
plt.close(fig)

# ── Fig: latency-budget compliance, point vs q90 vs oracle (dot plot) ────────
scen = list(r["scenarios"])
labels = ["urban\ndense", "urban\nsparse", "suburban", "highway", "rural"]
point = [r["scenarios"][s]["policies"]["point_ema"]["lat_ok_pct"] for s in scen]
q90 = [r["scenarios"][s]["policies"]["q90_ema"]["lat_ok_pct"] for s in scen]
oracle = [r["scenarios"][s]["policies"]["oracle"]["lat_ok_pct"] for s in scen]

fig, ax = plt.subplots(figsize=(3.5, 1.75), dpi=300)
x = range(len(scen))
# connector stems make the point→q90 improvement readable at a glance
for i in x:
    ax.plot([i, i], [point[i], q90[i]], color="#bbbbbb", lw=0.9, zorder=1)
ax.scatter(x, oracle, marker="^", s=26, facecolors="white",
           edgecolors="black", linewidths=0.9, label="oracle", zorder=3)
ax.scatter(x, point, marker="s", s=22, color="#8c8c8c",
           label="mean gate", zorder=2)
ax.scatter(x, q90, marker="o", s=24, color="black",
           label="q90 gate", zorder=3)

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=7.2)
ax.set_ylim(92.5, 100.9)
ax.set_ylabel("latency-budget\ncompliance (%)", fontsize=7.5)
ax.axhline(100, color="#000000", alpha=0.15, lw=0.7)
ax.grid(axis="y", color="#000000", alpha=0.08, lw=0.6)
ax.tick_params(length=2.5, width=0.6, labelsize=7.5)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.legend(loc="lower right", fontsize=6.8, ncol=3, handletextpad=0.15,
          columnspacing=0.8, borderaxespad=0.2)
fig.tight_layout(pad=0.35)
fig.savefig(FIGS / "compliance_gating.png", dpi=300)
plt.close(fig)

print("saved:", sorted(p.name for p in FIGS.iterdir()))
