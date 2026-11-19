"""Streamlit dashboard - Dynamic Software Effort Estimation from Customer Requirements.
Run:  streamlit run app.py
Edit / toggle / add requirements in the table: the whole pipeline re-executes automatically."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dynest import (EffortPipeline, RequirementTracker, SprintProblem, benchmark, demo_project, demo_team,
                    load_history, CLUSTERS)
from dynest.mamdani import INPUTS, trap
from dynest.data import FEATURES

st.set_page_config(page_title="DynEst - dynamic effort estimation", page_icon="📐", layout="wide")
INK, TEAL, RUST, VIOLET, MUTE = "#1F2A44", "#0F7C8A", "#C2410C", "#6B4E9B", "#8A93A6"
CCOL = {"UI-heavy": TEAL, "Backend-intensive": VIOLET, "Data-processing": RUST}
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;600;700&display=swap');
html, body, [class*="css"] {font-family: 'Public Sans', sans-serif;}
[data-testid="stMetric"] {background:#fff; border-left:4px solid #0F7C8A; padding:.6rem .9rem; border-radius:4px;}
h1 {font-weight:700; letter-spacing:-.01em;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training SOM, ANFIS and calibrating the fuzzy engine on historical projects...")
def get_pipeline():
    return EffortPipeline().fit(load_history())


def fresh_state():
    st.session_state.base_reqs = demo_project()
    st.session_state.tracker = None
    st.session_state.editor_ver = st.session_state.get("editor_ver", 0) + 1
    st.session_state.bench = None


def clean(df):
    """Make an edited table safe: fill blanks, auto-assign ids, remove duplicate ids."""
    d = df.copy().reset_index(drop=True)
    defaults = dict(title="New requirement", ui=3, backend=3, data=3, complexity=5, integration=5, clarity=5,
                    experience=5, size_points=5, priority=2, active=True)
    for c, v in defaults.items():
        d[c] = d[c].fillna(v)
    d["id"] = [i if isinstance(i, str) and i.strip() else f"N{k + 1:02d}" for k, i in enumerate(d["id"])]
    dup = d["id"].duplicated()
    if dup.any():
        st.warning(f"Duplicate ids ignored: {', '.join(d.loc[dup, 'id'])}")
        d = d[~dup]
    d["active"] = d["active"].astype(bool)
    return d


pipe = get_pipeline()
if "base_reqs" not in st.session_state:
    fresh_state()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Project settings")
    st.caption("Developers (specialty drives the sprint optimiser)")
    team = st.data_editor(demo_team(), num_rows="dynamic", key="team", width="stretch", column_config={
        "specialty": st.column_config.SelectboxColumn(options=CLUSTERS + ["Generalist"], required=True),
        "speed": st.column_config.NumberColumn(min_value=0.5, max_value=1.5, step=0.05, default=1.0)})
    team = team.dropna(subset=["name"]).fillna({"specialty": "Generalist", "speed": 1.0}).reset_index(drop=True)
    focus = st.slider("Focus factor (productive share of a day)", 0.5, 1.0, 0.8, 0.05)
    sprint_days = st.number_input("Sprint length (working days)", 5, 20, 10)
    rate = st.number_input("Cost per person-day (₹)", 500, 50000, 4000, 500)
    st.divider()
    threshold = st.slider("Re-optimise when effort drifts by ≥ (%)", 1, 30, 5) / 100
    auto_opt = st.toggle("Auto-run GA + PSO on significant change", True)
    iters = st.slider("Optimiser iterations", 30, 300, 100, 10)
    if st.button("Reset demo project", width="stretch"):
        fresh_state()
        st.rerun()

if team.empty:
    st.error("Add at least one developer in the sidebar.")
    st.stop()

# ---------------------------------------------------------------- header + editable backlog
st.title("Dynamic software effort estimation")
st.write("Edit the backlog below: change a rating, tick **active** off to drop a requirement, or add a row. "
         "The SOM → ANFIS → Mamdani pipeline re-executes at once and the sprint plan is re-optimised when the effort "
         "drifts past your threshold.")

cfg = {c: st.column_config.NumberColumn(min_value=0, max_value=10, step=1) for c in
       ["ui", "backend", "data", "complexity", "integration", "clarity", "experience"]}
cfg["size_points"] = st.column_config.NumberColumn("size (pts)", min_value=1, max_value=40, step=1)
cfg["priority"] = st.column_config.SelectboxColumn(options=[1, 2, 3], help="3 = must-have (scheduled first)")
cfg["active"] = st.column_config.CheckboxColumn("active", default=True)
edited = st.data_editor(st.session_state.base_reqs, num_rows="dynamic", width="stretch", column_config=cfg,
                        key=f"backlog_{st.session_state.editor_ver}", hide_index=True)
reqs = clean(edited)
if not reqs["active"].any():
    st.warning("All requirements are inactive - activate at least one.")
    st.stop()

tr = st.session_state.tracker
if tr is None:
    tr = RequirementTracker(pipe, team)
    st.session_state.tracker = tr
tr.team, tr.focus, tr.rate, tr.threshold = team, focus, float(rate), threshold
tr.sprint_days, tr.auto_optimize, tr.opt_iters = int(sprint_days), auto_opt, iters
with st.spinner("Re-estimating and re-optimising..."):
    tr.update(reqs)
snap = tr.snapshots[-1]
est = snap.estimate

c1, c2, c3, c4 = st.columns(4)
c1.metric("Effort (person-days)", f"{est.effort:.1f}", f"{snap.effort_delta:+.1f}" if snap.version else None,
          delta_color="inverse")
c1.caption(f"95% band ±{est.band:.1f}  ({est.low:.0f} - {est.high:.0f})")
c2.metric("Duration (working days)", f"{est.duration_days:.1f}", f"{snap.duration_delta:+.1f}" if snap.version else None,
          delta_color="inverse")
c2.caption(f"with {len(team)} developers; {est.duration_low:.0f} - {est.duration_high:.0f} d range")
c3.metric("Cost (₹)", f"{snap.cost:,.0f}", f"{snap.cost_delta:+,.0f}" if snap.version else None, delta_color="inverse")
c4.metric("Team for 4-sprint delivery", f"{est.team_for(4 * sprint_days)}", None)
c4.caption("developers needed to finish in four sprints")
with st.expander(f"Change log (version {snap.version})", expanded=False):
    for s in reversed(tr.snapshots[-8:]):
        flag = "re-optimised" if s.optimization else ("plan stale" if s.plan_stale else "")
        st.markdown(f"**v{s.version}** · {'; '.join(s.events)} · Δ {100 * s.pct_delta:+.1f}% · {flag}")

t_delta, t_som, t_exp, t_opt, t_model = st.tabs(
    ["Delta timeline", "SOM clusters", "Why this estimate", "Sprint optimisation", "Model quality"])

# ---------------------------------------------------------------- delta timeline
with t_delta:
    h = tr.history_frame()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.version, y=h.high, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h.version, y=h.low, fill="tonexty", fillcolor="rgba(15,124,138,.15)",
                             line=dict(width=0), name="95% band"))
    fig.add_trace(go.Scatter(x=h.version, y=h.effort, mode="lines+markers", name="effort", line=dict(color=TEAL, width=3),
                             text=h.events, hovertemplate="v%{x}<br>%{y:.1f} person-days<br>%{text}<extra></extra>"))
    fig.update_layout(title="Project effort after every requirement change", xaxis_title="version",
                      yaxis_title="person-days", height=380, xaxis=dict(dtick=1), margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(h.round(1), width="stretch", hide_index=True)

# ---------------------------------------------------------------- SOM
with t_som:
    t = est.table
    left, right = st.columns([3, 2])
    with left:
        hist = load_history().sample(250, random_state=0).copy()
        hist["cluster_name"] = [CLUSTERS[i] for i in pipe.som.transform(hist[FEATURES].values / 10)[1]]
        hist["source"] = "historical"
        cur = t[FEATURES + ["cluster_name"]].copy()
        cur["source"] = "current backlog"
        allp = pd.concat([hist[FEATURES + ["cluster_name", "source"]], cur], ignore_index=True)
        allp["size"] = np.where(allp.source == "current backlog", 12, 4)
        sm = px.scatter_matrix(allp, dimensions=["complexity", "integration", "ui", "backend", "data"],
                               color="cluster_name", color_discrete_map=CCOL, symbol="source",
                               symbol_map={"historical": "circle", "current backlog": "diamond"}, opacity=.8)
        sm.update_traces(diagonal_visible=False, marker=dict(size=5), showupperhalf=False)
        sm.update_layout(height=620, title="Requirements mapped to functional clusters", legend_title="")
        st.plotly_chart(sm, width="stretch")
    with right:
        som = pipe.som
        lab = som.neuron_labels().reshape(som.rows, som.cols)
        counts = np.bincount(t["som_neuron"], minlength=len(som.W)).reshape(som.rows, som.cols)
        hm = go.Figure(go.Heatmap(z=som.umatrix(), colorscale="Blues", showscale=False,
                                  text=[[f"{CLUSTERS[lab[r, c]][:4]}<br>{'●' * counts[r, c]}" for c in range(som.cols)]
                                        for r in range(som.rows)], texttemplate="%{text}", hoverinfo="skip"))
        hm.update_layout(title="SOM grid (shade = U-matrix distance, dots = current requirements)", height=380,
                         yaxis=dict(autorange="reversed", showticklabels=False), xaxis=dict(showticklabels=False))
        st.plotly_chart(hm, width="stretch")
        st.dataframe(t[["id", "title", "cluster_name", "som_neuron"]], hide_index=True, width="stretch")

# ---------------------------------------------------------------- explainability
with t_exp:
    t = est.table
    pick = st.selectbox("Requirement", t["id"] + " - " + t["title"])
    row = reqs[reqs["id"] == pick.split(" - ")[0]].iloc[0]
    r, fired, agg = pipe.explain(row)
    a, b = st.columns(2)
    with a:
        st.subheader("Activated Mamdani rules")
        fr = pd.DataFrame(fired)
        st.dataframe(fr, hide_index=True, width="stretch",
                     column_config={"firing_strength": st.column_config.ProgressColumn("firing", min_value=0, max_value=1)})
        st.markdown(f"**ANFIS complexity rating:** {r['complexity_rating']:.2f} / 10 &nbsp; "
                    f"**Cluster:** {r['cluster_name']}  \n"
                    f"**Defuzzified → calibrated:** {r['intensity']:.2f} person-days / point × {row['size_points']:g} pts "
                    f"= **{r['effort']:.1f} ± {1.96 * r['sigma']:.1f} person-days**")
    with b:
        y = pipe.fis.y
        fg = go.Figure(go.Scatter(x=y, y=agg, fill="tozeroy", line=dict(color=TEAL), name="aggregated output"))
        raw = float(np.sum(agg * y) / max(agg.sum(), 1e-12))
        fg.add_vline(x=raw, line_dash="dash", line_color=RUST, annotation_text=f"centroid {raw:.2f}")
        fg.update_layout(title="Aggregated fuzzy output (person-days per point)", height=330, margin=dict(t=50, b=10))
        st.plotly_chart(fg, width="stretch")
    mf = make_subplots(rows=1, cols=3, subplot_titles=["complexity (ANFIS)", "clarity", "experience"])
    xs = np.linspace(0, 10, 201)
    vals = dict(complexity=r["complexity_rating"], clarity=row["clarity"], experience=row["experience"])
    for k, (var, terms) in enumerate(INPUTS.items(), 1):
        for name, p in terms.items():
            mf.add_trace(go.Scatter(x=xs, y=trap(xs, *p), name=name, showlegend=False,
                                    hovertemplate=name + "<extra></extra>"), row=1, col=k)
        mf.add_vline(x=vals[var], line_dash="dash", line_color=RUST, row=1, col=k)
    mf.update_layout(height=260, margin=dict(t=40, b=10), title="Input membership functions (dashed = this requirement)")
    st.plotly_chart(mf, width="stretch")
    st.subheader("Rules active across the whole backlog")
    st.dataframe(est.rule_summary.head(12).round(3), hide_index=True, width="stretch")
    st.caption(f"Confidence band: per-requirement spread σ_log = {pipe.sigma_log:.2f} measured on held-out history, "
               "inflated for vague requirements, combined across requirements as independent errors plus a 3% "
               "systematic term; shown as ±1.96σ.")

# ---------------------------------------------------------------- optimisation
with t_opt:
    if not snap.optimization:
        if snap.plan_stale:
            st.warning("Requirements changed but the effort drift is below your threshold, so the current plan "
                       "may be out of date.")
        if st.button("Run GA + PSO now", type="primary"):
            tr.optimize(snap)
            st.rerun()
    if tr.last_results:
        res, prob = tr.last_results, tr.last_problem
        if snap.plan_stale and snap.optimization == {}:
            st.info("Showing the most recent optimised plan (older than the latest change).")
        cc = go.Figure()
        for name, col in (("GA", TEAL), ("PSO", RUST)):
            cc.add_trace(go.Scatter(y=res[name].history, name=name, line=dict(color=col, width=3)))
        cc.update_layout(title="Fitness convergence (lower is better)", xaxis_title="iteration", yaxis_title="fitness",
                         height=330, margin=dict(t=50, b=10))
        st.plotly_chart(cc, width="stretch")
        mt = pd.DataFrame({n: dict(fitness=r.fitness, seconds=r.time_s, converged_at_iter=r.conv_iter,
                                   leveling_efficiency=r.metrics["leveling_efficiency"],
                                   utilization=r.metrics["utilization"], overload_days=r.metrics["overload_days"],
                                   sprints_used=r.metrics["sprints_used"], tasks_moved=r.metrics["tasks_moved"])
                           for n, r in res.items()}).round(3)
        st.dataframe(mt, width="stretch")
        best = min(res.values(), key=lambda r: r.fitness)
        st.subheader(f"Sprint plan (best: {best.name})")
        plan = prob.plan(best.genes)
        load = plan.pivot_table(index="sprint", columns="developer", values="effective_days", aggfunc="sum").reindex(
            columns=prob.dev_names).fillna(0)
        txt = plan.groupby(["sprint", "developer"])["id"].apply(", ".join).unstack().reindex(columns=prob.dev_names).fillna("")
        hm = go.Figure(go.Heatmap(z=load.values, x=load.columns, y=[f"Sprint {s}" for s in load.index],
                                  text=txt.reindex(load.index).values, texttemplate="%{text}", colorscale="Teal",
                                  zmin=0, zmax=prob.cap * 1.2, colorbar=dict(title="days")))
        hm.update_layout(height=120 + 70 * len(load), yaxis=dict(autorange="reversed"), margin=dict(t=20, b=10))
        st.plotly_chart(hm, width="stretch")
        st.caption(f"Capacity per developer per sprint = {prob.cap:.1f} productive days. Requirements larger than "
                   "~75% of that are split into chunks (R06#1, R06#2 ...).")
        st.dataframe(plan.sort_values(["sprint", "developer"]), hide_index=True, width="stretch")
        st.subheader("GA vs PSO benchmark")
        runs = st.slider("Independent runs (different seeds)", 3, 20, 5)
        if st.button("Run benchmark"):
            with st.spinner("Benchmarking..."):
                st.session_state.bench = benchmark(prob, runs=runs, iters=iters)
        if st.session_state.bench is not None:
            summ, _, raw = st.session_state.bench
            st.dataframe(summ.T, width="stretch")
            bx = px.box(raw, x="algorithm", y="fitness", color="algorithm", points="all",
                        color_discrete_map={"GA": TEAL, "PSO": RUST}, title="Final fitness over independent runs")
            bx.update_layout(height=320, showlegend=False)
            st.plotly_chart(bx, width="stretch")

# ---------------------------------------------------------------- model quality
with t_model:
    st.subheader("Held-out accuracy (20% of the historical database)")
    st.dataframe(pipe.metrics.round(3), width="stretch")
    st.caption("MMRE = mean magnitude of relative error (lower is better); PRED25 = share of estimates within 25% of "
               "actual effort (higher is better). The baseline is a COCOMO-style linear regression on the same attributes.")
    a, b = st.columns(2)
    hh = np.array(pipe.anfis.history)
    f1 = go.Figure([go.Scatter(y=hh[:, 0], name="train RMSE", line=dict(color=TEAL)),
                    go.Scatter(y=hh[:, 1], name="validation RMSE", line=dict(color=RUST))])
    f1.update_layout(title="ANFIS hybrid learning (0-1 scale)", xaxis_title="epoch", height=300, margin=dict(t=50))
    a.plotly_chart(f1, width="stretch")
    f2 = go.Figure(go.Scatter(y=pipe.som.qe_history, line=dict(color=VIOLET)))
    f2.update_layout(title="SOM quantisation error", xaxis_title="epoch", height=300, margin=dict(t=50))
    b.plotly_chart(f2, width="stretch")
    st.write(f"ANFIS test RMSE on the 0-10 complexity rating: **{pipe.anfis_rmse_test:.2f}**; "
             f"SOM test quantisation error: **{pipe.som_qe:.3f}**.")
