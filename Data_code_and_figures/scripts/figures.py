"""Publication figures for the manuscript (300/600 dpi PNG)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "out"; FIG = ROOT / "fig"; FIG.mkdir(exist_ok=True)
for f in fm.findSystemFonts():
    if "Carlito" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Carlito", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.linewidth": 0.8, "savefig.dpi": 600, "figure.dpi": 150})
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
R = json.loads((OUT / "results.json").read_text())
s = pd.read_csv(OUT / "scored.csv"); n = len(s)

LAB = {"P_safe": "World is pretty safe", "P_dangerous": "World is dangerous (R)", "P_harmless": "Most things harmless, safe",
       "P_good": "Most things are good", "P_cutthroat": "World is cut-throat, competitive (R)", "P_worse": "Things tend to get worse (R)",
       "P_beautyugly": "More beauty than ugliness", "P_worth": "Most things not worth exploring (R)", "P_boring": "Most things boring (R)",
       "P_abundant": "World is abundant", "P_fascinating": "World is fascinating", "P_dull": "World is dull (R)",
       "P_beautyaround": "Beauty always around us", "P_helpme": "Events happen to help me", "P_meant": "What happens is meant to",
       "P_lackpurpose": "Events lack bigger purpose (R)", "P_needsme": "Universe needs me", "P_reason": "Everything happens for a reason"}
GROUPS = [("Safe", ["P_safe", "P_dangerous", "P_harmless", "P_good", "P_cutthroat", "P_worse"]),
          ("Enticing", ["P_beautyugly", "P_worth", "P_boring", "P_abundant", "P_fascinating", "P_dull", "P_beautyaround"]),
          ("Alive", ["P_helpme", "P_meant", "P_lackpurpose", "P_needsme", "P_reason"])]

def ci(x):
    x = np.asarray(x); m = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return m, m - h, m + h

# ---------------------------------------------------------------- Figure 1: items vs reference values
ni = pd.read_csv(OUT / "norm_item_comparison.csv").set_index("item")
fig, ax = plt.subplots(figsize=(6.5, 6.4))
y = 0; yt, yl, heads = [], [], []
for g, items in GROUPS:
    items = sorted(items, key=lambda c: ni.loc[c, "diff"])
    heads.append((y - 0.2, g))
    y += 0.9
    for c in items:
        m, lo, hi = ci(s[c])
        ax.plot([lo, hi], [y, y], color=BLUE, lw=1.6, solid_capstyle="round", zorder=2)
        ax.scatter([m], [y], s=34, color=BLUE, edgecolor="white", linewidth=1.2, zorder=3)
        ax.scatter([ni.loc[c, "norm_M"]], [y], s=34, marker="D", facecolor="white", edgecolor=MUTED, linewidth=1.1, zorder=3)
        d = ni.loc[c, "d"]; sig = ni.loc[c, "p_holm"] < .05
        ax.text(5.18, y, f"{d:+.2f}{'*' if sig else ''}", va="center", ha="left", fontsize=8.5, color=INK if sig else MUTED)
        yt.append(y); yl.append(LAB[c]); y += 1
    y += 0.5
for yy, g in heads:
    ax.text(-0.05, yy + 0.35, g, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10, fontweight="bold", color=INK)
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8.8); ax.invert_yaxis()
ax.set_xlim(0, 5); ax.set_xticks(range(6))
ax.set_xlabel("Item score (0 = strongly disagree with the positive pole, 5 = strongly agree)")
ax.grid(axis="x", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
ax.text(5.18, -0.9, "d", ha="left", fontsize=9, style="italic", color=INK)
ax.axvline(2.5, color=GREY, lw=0.8, ls=(0, (3, 3)), zorder=1)
leg = [Line2D([0], [0], marker="o", color=BLUE, lw=1.6, markersize=6, markeredgecolor="white", label=f"Present sample (N = {n}), mean and 95% CI"),
       Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=MUTED, markersize=6,
              label="Reference mean, Clifton & Yaden (2021) Study 2 (N = 5,171)")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.42, -0.085), ncol=1, frameon=False, fontsize=8.6)
fig.tight_layout(); fig.savefig(FIG / "fig1_items_vs_reference.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- Figure 2: domain vs general
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 3.3), gridspec_kw={"width_ratios": [1.05, 1]})
prim = [("Safe", "Eng_Safe", ["P_safe", "P_good", "P_harmless"]),
        ("Enticing", "Eng_Enticing", ["P_beautyugly", "P_abundant", "P_fascinating", "P_beautyaround"]),
        ("Alive", "Eng_Alive", ["P_helpme", "P_meant", "P_needsme", "P_reason"])]
for i, (g, e, fwd) in enumerate(prim):
    vals = [(s[g], BLUE, "o", "full"), (s[fwd].mean(axis=1), BLUE, "o", "none"), (s[e], ORANGE, "s", "full")]
    for j, (v, col, mk, fill) in enumerate(vals):
        m, lo, hi = ci(v); x = i + (j - 1) * 0.24
        a.plot([x, x], [lo, hi], color=col, lw=1.6, solid_capstyle="round")
        a.scatter([x], [m], s=38, marker=mk, color=col if fill == "full" else "white", edgecolor=col, linewidth=1.4, zorder=3)
    dz = R["domain_vs_general"][f"{e}-{g}"]["dz"]; p = R["domain_vs_general"][f"{e}-{g}"]["p"]
    a.text(i, 4.35, f"$d_z$ = {dz:.2f}", ha="center", fontsize=8.5, color=INK if p < .05 else MUTED)
a.set_xticks(range(3)); a.set_xticklabels(["Safe", "Enticing", "Alive"]); a.set_ylim(1.5, 4.6); a.set_xlim(-0.5, 2.5)
a.axhline(2.5, color=GREY, lw=0.8, ls=(0, (3, 3))); a.set_ylabel("Scale score (0–5), mean and 95% CI")
a.grid(axis="y", color="#e6e5e1", lw=0.6); a.set_axisbelow(True)
a.set_title("A. General vs. engineering-specific", loc="left", fontsize=10, fontweight="bold")
leg = [Line2D([0], [0], marker="o", color=BLUE, lw=0, markersize=6, label="General primal (PI-18)"),
       Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=BLUE, markersize=6, label="General, positively keyed items only"),
       Line2D([0], [0], marker="s", color=ORANGE, lw=0, markersize=6, label="Engineering-specific")]
fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=8.3, handletextpad=0.3)
rngj = np.random.default_rng(3)
xs = s["Safe"] + rngj.uniform(-.03, .03, n); ys = s["Eng_Safe"] + rngj.uniform(-.03, .03, n)
b.scatter(xs, ys, s=16, color=ORANGE, alpha=.75, edgecolor="white", linewidth=0.5)
b.axvline(2.5, color=GREY, lw=0.8, ls=(0, (3, 3))); b.axhline(2.5, color=GREY, lw=0.8, ls=(0, (3, 3)))
q = {"ul": int(((s.Safe < 2.5) & (s.Eng_Safe > 2.5)).sum()), "ur": int(((s.Safe > 2.5) & (s.Eng_Safe > 2.5)).sum()),
     "ll": int(((s.Safe < 2.5) & (s.Eng_Safe < 2.5)).sum()), "lr": int(((s.Safe > 2.5) & (s.Eng_Safe < 2.5)).sum())}
ties = n - sum(q.values())
b.text(0.1, 4.85, f"n = {q['ul']}", fontsize=8.5, color=INK, fontweight="bold"); b.text(4.9, 4.85, f"n = {q['ur']}", fontsize=8.5, ha="right", color=MUTED)
b.text(0.1, 0.1, f"n = {q['ll']}", fontsize=8.5, color=MUTED); b.text(4.9, 0.1, f"n = {q['lr']}", fontsize=8.5, ha="right", color=MUTED)
b.text(0.1, 4.5, "world unsafe,\nengineering safe", fontsize=7.5, color=MUTED, va="top")
z = np.polyfit(s.Safe, s.Eng_Safe, 1); xx = np.linspace(0, 5, 10)
b.plot(xx, np.polyval(z, xx), color=INK, lw=1.0, alpha=.6)
b.set_xlim(0, 5); b.set_ylim(0, 5); b.set_xlabel("General Safe (PI-18)"); b.set_ylabel("Engineering-specific Safe")
b.text(4.9, 1.0, f"r = {R['cross']['Safe~Eng_Safe']['r']:.2f}", ha="right", fontsize=8.5, color=INK)
b.set_title("B. Safe: person-level pattern", loc="left", fontsize=10, fontweight="bold")
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / "fig2_domain_vs_general.png", bbox_inches="tight"); plt.close(fig)
json.dump(dict(q, ties=ties), open(OUT / "fig2_quadrants.json", "w"))

# ---------------------------------------------------------------- Figure 3: item correlation heatmap
order = [i for _, it in GROUPS for i in it] + [f"ES{i}" for i in range(1, 7)] + [f"EE{i}" for i in range(1, 7)] + [f"EA{i}" for i in range(1, 7)]
C = s[order].corr().values
cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f2f2f0", ORANGE])
fig, ax = plt.subplots(figsize=(6.4, 5.6))
im = ax.imshow(C, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
bounds = [0, 6, 13, 18, 24, 30, 36]; names = ["G-Safe", "G-Enticing", "G-Alive", "E-Safe", "E-Enticing", "E-Alive"]
for k in range(6):
    lo, hi = bounds[k] - .5, bounds[k + 1] - .5
    ax.add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo, fill=False, ec=INK, lw=1.0))
ax.axhline(17.5, color=INK, lw=1.6); ax.axvline(17.5, color=INK, lw=1.6)
mids = [(bounds[k] + bounds[k + 1] - 1) / 2 for k in range(6)]
ax.set_xticks(mids); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8.5)
ax.set_yticks(mids); ax.set_yticklabels(names, fontsize=8.5)
ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Pearson r (reverse-keyed items rescored)", fontsize=8.5); cb.outline.set_visible(False)
def blockmean(r0, r1, c0, c1, same):
    M = C[r0:r1, c0:c1]
    return M[np.triu_indices(r1 - r0, 1)].mean() if same else M.mean()
gg = np.mean([C[i, j] for i in range(18) for j in range(i + 1, 18)]); ee = np.mean([C[i, j] for i in range(18, 36) for j in range(i + 1, 36)])
ge = C[:18, 18:].mean()
ax.set_title("General (G) items: mean r = {:.2f}   Engineering (E) items: mean r = {:.2f}   G × E: mean r = {:.2f}".format(gg, ee, ge),
             fontsize=8.3, color=MUTED, pad=8)
fig.tight_layout(); fig.savefig(FIG / "fig3_item_correlations.png", bbox_inches="tight"); plt.close(fig)
json.dump(dict(mean_r_general=gg, mean_r_eng=ee, mean_r_cross=ge), open(OUT / "fig3_block_means.json", "w"))

# ---------------------------------------------------------------- Figure 4: acquiescence correlations
S2 = R["S2"]
rows = [("General Safe", "Safe", BLUE), ("General Enticing", "Enticing", BLUE), ("General Alive", "Alive", BLUE), ("General Good", "Good", BLUE),
        ("Engineering Safe", "Eng_Safe", ORANGE), ("Engineering Enticing", "Eng_Enticing", ORANGE), ("Engineering Alive", "Eng_Alive", ORANGE),
        ("Engineering total", "Eng_Total", ORANGE)]
fig, ax = plt.subplots(figsize=(5.6, 3.0))
for i, (lab, k, col) in enumerate(rows):
    v = S2[f"r_ARS_SE~{k}"]; yy = i + (0.5 if i >= 4 else 0)
    ax.plot(v["CI"], [yy, yy], color=col, lw=1.6, solid_capstyle="round")
    ax.scatter([v["r"]], [yy], s=36, color=col, marker="o" if i < 4 else "s", edgecolor="white", linewidth=1.1, zorder=3)
    ax.text(0.62, yy, f"{v['r']:+.2f}", va="center", fontsize=8.5, color=INK if v["p"] < .05 else MUTED)
ax.set_yticks([i + (0.5 if i >= 4 else 0) for i in range(8)]); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.axvline(0, color=GREY, lw=0.9); ax.set_xlim(-0.4, 0.6)
ax.set_xlabel("Correlation with content-balanced acquiescence index (95% CI)")
ax.grid(axis="x", color="#e6e5e1", lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(FIG / "fig4_acquiescence.png", bbox_inches="tight"); plt.close(fig)
print("figures done")
