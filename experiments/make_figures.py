"""Regenerate every paper figure from experiments/results/*.csv alone (no model is refitted).

Output: experiments/results/figures/<name>.pdf and .png.  A figure whose inputs are missing is skipped
with a message.  Colour roles: proposed model = blue, its ablations = orange, every baseline = neutral grey
(validated palette; identity is also carried by direct labels, never by colour alone).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dynest import stats  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
OUT = RES / "figures"
BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
    "legend.frameon": False,
})


def role(name):
    n = str(name)
    if "proposed" in n:
        return "proposed"
    if n.startswith(("A1", "A2", "A3", "A4", "A5", "NF -", "ANFIS only")):
        return "ablation"
    return "baseline"


COL = dict(proposed=BLUE, ablation=ORANGE, baseline=GREY)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    PAPER = ROOT / "paper" / "figures"                       # self-contained manuscript folder (Overleaf)
    PAPER.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER / f"{name}.pdf")
    plt.close(fig)
    print("wrote", (OUT / f"{name}.pdf").relative_to(ROOT))


def need(*files):
    missing = [f for f in files if not (RES / f).exists()]
    if missing:
        print("skip (missing):", ", ".join(missing))
    return not missing


# ---------------------------------------------------------------------------------------------- ranks
def rank_panel(ax, table, lower, title, xlabel):
    out = stats.compare(table, lower_is_better=lower)
    r = pd.Series(out["mean_rank"]).sort_values()
    y = np.arange(len(r))[::-1]
    for yi, (m, v) in zip(y, r.items()):
        ax.plot([1, v], [yi, yi], color=GRID, lw=1.2, zorder=1)
        ax.scatter(v, yi, s=34, color=COL[role(m)], zorder=3, edgecolor="white", linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(r.index)
    for lab, m in zip(ax.get_yticklabels(), r.index):
        if role(m) == "proposed":
            lab.set_fontweight("bold")
    cd, best = out["cd_nemenyi"], r.iloc[0]
    ax.plot([best, best + cd], [len(r) - 0.3] * 2, color=INK, lw=2, solid_capstyle="butt", clip_on=False)
    ax.text(best + cd / 2, len(r) - 0.05, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_ylim(-0.7, len(r) + 0.4)
    ax.set_xlim(0.5, len(r) + 0.5)
    ax.grid(axis="y", visible=False)
    p = out.get("friedman_p", np.nan)
    ax.set_title(f"{title}\nFriedman p = {p:.2g}, n = {out['n_units']}", loc="left")
    ax.set_xlabel(xlabel)


def fig_ranks():
    if not need("e1_accuracy_raw.csv", "e2_unseen_projects.csv", "e8_summary.csv"):
        return
    e1 = pd.read_csv(RES / "e1_accuracy_raw.csv")
    e1 = e1[e1.project_id.astype(str) != "ALL"]
    e1 = e1[~e1.model.map(role).eq("ablation")]
    t1 = e1.groupby(["project_id", "model"]).MAE.mean().unstack()
    e2 = pd.read_csv(RES / "e2_unseen_projects.csv")
    t2 = e2.pivot_table(index="project_id", columns="model", values="MAE")
    e8 = pd.read_csv(RES / "e8_summary.csv")
    e8 = e8[~e8.model.str.startswith("Expert") & ~e8.model.map(role).eq("ablation")]
    t8 = e8.pivot_table(index="dataset", columns="model", values="SA").dropna(axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(wspace=0.75))
    rank_panel(axes[0], t1, True, "E1 TAWOS, later data (MAE)", "mean rank over projects (1 = best)")
    rank_panel(axes[1], t2, True, "E2 TAWOS, unseen projects (MAE)", "mean rank over held-out projects")
    rank_panel(axes[2], t8, False, "E8 classic FP/COCOMO datasets (SA)", "mean rank over datasets")
    save(fig, "fig_ranks")


# ---------------------------------------------------------------------------------------------- uncertainty
def fig_intervals():
    if not need("e1_accuracy_raw.csv", "e8_summary.csv"):
        return
    e1 = pd.read_csv(RES / "e1_accuracy_raw.csv")
    a = e1[e1.project_id.astype(str) == "ALL"].groupby("model")[["coverage", "rel_width"]].mean().dropna()
    e8 = pd.read_csv(RES / "e8_summary.csv").dropna(subset=["coverage"])
    b = e8.groupby("model")[["coverage", "rel_width"]].median()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(wspace=0.3))
    for ax, d, title in ((axes[0], a, "E1 TAWOS requirements (pooled test)"),
                         (axes[1], b, "E8 classic datasets (median over datasets)")):
        for yy in (0.92, 0.98):
            ax.axhline(yy, color=GREY, lw=0.6, ls=":", zorder=1)
        ax.axhline(0.95, color=INK2, lw=0.8, ls="--", zorder=1)
        for m, r in d.iterrows():
            ax.scatter(r.rel_width, r.coverage, s=36, color=COL[role(m)], edgecolor="white", linewidth=1.5, zorder=3)
        for m in d.index[d.index.map(role) == "proposed"]:
            ax.annotate(m.replace(" (proposed)", ""), (d.loc[m, "rel_width"], d.loc[m, "coverage"]),
                        xytext=(6, 6), textcoords="offset points", fontsize=7.5, fontweight="bold")
        ax.set_ylim(0.90, 1.0)
        ax.set_xlabel("median relative interval width, (high - low) / estimate")
        ax.set_ylabel("coverage of the 95% interval")
        ax.set_title(title, loc="left")
        ax.text(0.99, 0.955, "target band 0.95 ± 0.03", transform=ax.get_yaxis_transform(), ha="right",
                fontsize=7, color=INK2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in
               (("proposed", BLUE), ("ablation", ORANGE), ("baseline", GREY))]
    axes[1].legend(handles=handles, loc="lower right")
    save(fig, "fig_intervals")


# ---------------------------------------------------------------------------------------------- ablations
def fig_ablation():
    if not need("e1_accuracy_raw.csv", "e8_summary.csv"):
        return
    e1 = pd.read_csv(RES / "e1_accuracy_raw.csv")
    e1 = e1[e1.project_id.astype(str) == "ALL"].groupby("model").MAE.mean()
    full = e1[[m for m in e1.index if role(m) == "proposed"][0]]
    d1 = ((e1[e1.index.map(role) == "ablation"] / full - 1) * 100).sort_values()
    e8 = pd.read_csv(RES / "e8_summary.csv")
    t = e8.pivot_table(index="dataset", columns="model", values="SA")
    prop = [c for c in t if role(c) == "proposed"][0]
    abl = [c for c in t if role(c) == "ablation"]
    d8 = t[abl].sub(t[prop], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw=dict(wspace=0.6))
    ax = axes[0]
    ax.barh(range(len(d1)), d1.values, color=ORANGE, height=0.55)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(d1)))
    ax.set_yticklabels(d1.index)
    for i, v in enumerate(d1.values):
        ax.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right",
                fontsize=7.5, color=INK2)
    ax.set_xlabel("change in pooled test MAE vs full DynEst-R (%, > 0 = worse)")
    ax.set_title("E1 TAWOS: removing one component", loc="left")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    order = d8.mean().sort_values().index
    for i, m in enumerate(order):
        ax.scatter(d8[m], np.full(len(d8), i) + np.linspace(-0.15, 0.15, len(d8)), s=12, color=ORANGE, alpha=0.6,
                   edgecolor="none")
        ax.scatter(d8[m].mean(), i, s=60, marker="D", color=INK, zorder=3)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("SA difference vs full model per dataset (> 0 = ablation better); diamond = mean")
    ax.set_title("E8 classic datasets: ablations", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_ablation")


# ---------------------------------------------------------------------------------------------- proxy validity
def fig_proxy():
    if not need("e0_proxy_validity_projects.csv"):
        return
    P = pd.read_csv(RES / "e0_proxy_validity_projects.csv")
    names = {"effort_h": "proxy: WIP-adjusted active hours (used)", "active_bh": "proxy: active hours, no WIP sharing",
             "In_Progress_Minutes": "TAWOS In_Progress_Minutes", "Total_Effort_Minutes": "TAWOS Total_Effort_Minutes",
             "Resolution_Time_Minutes": "cycle time (creation to resolution)", "Story_Point": "story points"}
    fig, ax = plt.subplots(figsize=(7, 3.2))
    keys = list(names)[::-1]
    for i, k in enumerate(keys):
        v = P[f"rho_{k}"].dropna().values
        c = BLUE if k == "effort_h" else GREY
        ax.scatter(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), s=14, color=c,
                   alpha=0.7, edgecolor="none")
        ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
        ax.text(0.98, i, f"median {np.median(v):.2f} (n = {len(v)})", transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([names[k] for k in keys])
    ax.set_xlim(-0.2, 1.2)
    ax.set_xlabel("Spearman correlation with logged work time (Timespent), one dot per project")
    ax.set_title("E0 effort-proxy validity: 15 TAWOS projects with logged time", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_proxy")


# ---------------------------------------------------------------------------------------------- re-planning
def fig_replanning():
    if not need("e5_policies.csv"):
        return
    E = pd.read_csv(RES / "e5_policies.csv")
    base = E[E.policy == "static"].set_index(["origin", "episode"])
    rows = []
    for p, g in E.groupby("policy"):
        g = g.set_index(["origin", "episode"])
        d = g.weighted_completion - base.weighted_completion.reindex(g.index)
        rows.append(dict(policy=p, d_completion=d.mean(), churn=g.churn_total.mean(), replans=g.replans.mean()))
    R = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    main = {"static", "sprint-boundary", "learned", "tuned-tau", "tau=0", "hybrid(tau=0.05)"}
    for _, r in R.iterrows():
        is_main = r.policy in main
        c = BLUE if r.policy == "learned" else (INK if is_main else GREY)
        ax.scatter(r.churn, r.d_completion, s=40 if is_main else 18, color=c, edgecolor="white", linewidth=1.5, zorder=3)
        if is_main:
            off = {"sprint-boundary": (-8, 8, "right"), "hybrid(tau=0.05)": (0, 10, "center")}.get(r.policy, (6, 4, "left"))
            ax.annotate(r.policy, (r.churn, r.d_completion), xytext=off[:2], textcoords="offset points", fontsize=7.5,
                        ha=off[2])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("mean plan churn per episode")
    ax.set_ylabel("weighted completion vs static (sprints, < 0 = earlier)")
    ax.set_title("E5 re-planning policies on 88 replayed sprint windows\n(simulation-relative; grey = other tau values)",
                 loc="left")
    save(fig, "fig_replanning")


if __name__ == "__main__":
    for f in (fig_ranks, fig_intervals, fig_ablation, fig_proxy, fig_replanning):
        try:
            f()
        except Exception as e:                                   # one broken input must not stop the rest
            print(f"FAILED {f.__name__}: {type(e).__name__}: {e}")
