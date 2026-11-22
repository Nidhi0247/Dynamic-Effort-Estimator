"""DynEst-R dashboard: replay a real held-out project window and watch estimation + re-planning react.

    streamlit run app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments"))
from common import RESULTS, dataset, efficiency_models, fitted_model, origin_splits, replay_builder  # noqa: E402
from dynest.scheduling import Problem, run_nsga2, OBJECTIVES  # noqa: E402
from dynest.sim import Episode, SprintBoundary, Static, Threshold  # noqa: E402
from dynest.tracker import ReplayTracker  # noqa: E402

st.set_page_config(page_title="DynEst-R", layout="wide")


@st.cache_resource(show_spinner="Loading real data and the trained model ...")
def load(backend):
    ds = dataset()
    f, tr, va, te = origin_splits()[0]
    m = fitted_model(0, tr, va, backend)
    eff_plan, eff_true = efficiency_models(m, tr, va, te)
    return ds, te, m, eff_plan, eff_true, replay_builder()


@st.cache_resource(show_spinner="Replaying the project window ...")
def replay(p, si, policy_name, tau, opt, backend):
    ds, te, m, eff_plan, eff_true, rb = load(backend)
    w = te[te.project_id == p]
    start = rb.episode_starts(p, w.resolved_at.min(), w.resolved_at.max())[si]
    ep, info = rb.build(p, start, 4, m, eff_plan, eff_true)
    pol = {"Threshold (tau)": Threshold(tau), "Sprint boundary": SprintBoundary(), "Static plan": Static()}[policy_name]
    trk = ReplayTracker(ep, m, pol, opt, opt_kw=dict(gens=40, pop=40)).run()
    return ep, info, trk


backend = st.sidebar.selectbox("Text encoder", ["minilm", "tfidf"])
ds, te, m, eff_plan, eff_true, rb = load(backend)
projects = sorted(te.project_id.unique(), key=lambda p: -int((te.project_id == p).sum()))
p = st.sidebar.selectbox("Project (held-out window)", projects, format_func=lambda x: f"{ds.keys.get(x, x)} ({x})")
w = te[te.project_id == p]
starts = rb.episode_starts(p, w.resolved_at.min(), w.resolved_at.max())
if not starts:
    st.warning("No sprint starts in this project's held-out window.")
    st.stop()
si = st.sidebar.selectbox("Window (real sprint start / team board)", range(len(starts)),
                          format_func=lambda i: f"{pd.Timestamp(starts[i][0]).date()} | {starts[i][1]}")
policy_name = st.sidebar.radio("Re-planning policy", ["Threshold (tau)", "Sprint boundary", "Static plan"])
tau = st.sidebar.slider("Trigger threshold tau (%)", 0, 30, 5) / 100
opt = st.sidebar.selectbox("Optimiser", ["ga", "pso", "nsga2", "mopso", "greedy"])
ep, info, trk = replay(p, si, policy_name, tau, opt, backend)
h = trk.history_frame()
o = trk.outcome()

c = st.columns(6)
c[0].metric("Scope effort (h)", f"{h.effort.iloc[-1]:,.0f}", f"{h.effort.iloc[-1] - h.effort.iloc[0]:+,.0f}")
c[1].metric("95% interval (h)", f"{h.low.iloc[-1]:,.0f} - {h.high.iloc[-1]:,.0f}")
c[2].metric("Change events", f"{len(ep.events)}")
c[3].metric("Re-optimisations", f"{o['replans']}")
c[4].metric("Total churn", f"{o['churn_total']:.2f}")
c[5].metric("Weighted completion (sprints)", f"{o['weighted_completion']:.2f}")

tabs = st.tabs(["Delta timeline", "SOM clusters", "Why this estimate", "Sprint plan & Pareto", "Model quality"])
with tabs[0]:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.version, y=h.high, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=h.version, y=h.low, fill="tonexty", line=dict(width=0), name="95% interval"))
    fig.add_trace(go.Scatter(x=h.version, y=h.effort, mode="lines+markers", name="re-estimated scope effort",
                             marker=dict(color=np.where(h.decision == "replan", "crimson", "steelblue"))))
    fig.update_layout(xaxis_title="change event #", yaxis_title="hours", height=420)
    st.plotly_chart(fig, width="stretch")
    st.caption("Red markers: the policy re-optimised the plan; blue: repair-only insertion.")
    st.dataframe(h.round(2), width="stretch", height=320)
with tabs[1]:
    U = m.som.umatrix()
    g = m.som.rows
    st.plotly_chart(go.Figure(go.Heatmap(z=U, colorscale="Blues")).update_layout(title="U-matrix", height=380),
                    width="stretch")
    st.dataframe(pd.DataFrame(dict(neuron=range(len(m.som.names)), name=m.som.names,
                                   training_issues=m.som.counts, prior_log_h_per_sp=np.round(m.som.prior, 3))))
    st.caption(f"Component topics learnt from the data: {m.aux.topic_names if m.aux else []}")
with tabs[2]:
    fr = info["fr"]
    pick = st.selectbox("Requirement", list(fr.index), format_func=lambda i: f"{fr.issue_key[i]}")
    feats = rb.ctx.features_as_of(pd.Series([info["entry"][pick]], index=[pick]))
    ex = m.explain(feats)
    pr = m.predict(feats).iloc[0]
    st.write(f"**{fr.issue_key[pick]}**: {str(feats.text.iloc[0])[:300]}")
    st.write(f"Cluster **{ex['cluster']}** | ANFIS complexity {ex['c_hat']:.2f} | clarity {ex['clarity_hat']:.2f} | "
             f"estimate **{pr.hours:.1f} h** (95% interval {pr.low:.1f} to {pr.high:.1f}) | measured effort {fr.effort_h[pick]:.1f} h")
    if ex.get("rules"):
        r = pd.DataFrame(ex["rules"])
        st.plotly_chart(go.Figure(go.Bar(x=r.firing_strength, y=r.rule, orientation="h")).update_layout(
            height=320, yaxis=dict(autorange="reversed"), title="Fired learned rules"), width="stretch")
        st.plotly_chart(go.Figure(go.Scatter(x=ex["ygrid"], y=ex["aggregated"], fill="tozeroy")).add_vline(
            x=ex["crisp"]).update_layout(height=260, xaxis_title="log hours per story point", title="Aggregated output"),
            width="stretch")
    st.dataframe(m.fis.rule_table().sort_values("support", ascending=False), width="stretch", height=260)
with tabs[3]:
    keys = info["fr"].issue_key
    snaps = [s for s in trk.snapshots if s.plan]
    v = st.select_slider("Plan after change event #", options=[s.version for s in snaps],
                         value=snaps[-1].version) if len(snaps) > 1 else (snaps[0].version if snaps else 0)
    snap = next((s for s in snaps if s.version == v), None)
    if snap is None:
        st.info("No open work in this window.")
    else:
        plan = pd.DataFrame([dict(issue=keys.get(int(i), i), sprint=s + 1, developer=ep.dev_names[d])
                             for i, (s, d) in snap.plan.items()])
        st.caption(f"Event: {snap.event} | decision: {snap.decision} | churn {snap.churn:.2f}")
        st.dataframe(plan.sort_values(["sprint", "developer"]), width="stretch", height=300)
    if st.button("Show the Pareto front of the current re-planning instance"):
        e2 = Episode(ep, Static())
        inst = e2._instance(repair=False)
        prob = Problem(inst)
        r = run_nsga2(prob, gens=60)
        F = pd.DataFrame(r.front_F, columns=OBJECTIVES)
        st.plotly_chart(go.Figure(go.Splom(dimensions=[dict(label=c, values=F[c]) for c in OBJECTIVES])).update_layout(
            height=520, title="NSGA-II Pareto front (cost, schedule, workload, churn)"), width="stretch")
with tabs[4]:
    for name in ("e1_accuracy_summary.csv", "e5_tau_sensitivity.csv", "e6_optimisers.csv", "e7_preferences.csv"):
        f = RESULTS / name
        if f.exists():
            st.subheader(name)
            st.dataframe(pd.read_csv(f).round(3), width="stretch")
    if not (RESULTS / "e1_accuracy_summary.csv").exists():
        st.info("Run the experiments (see README) to populate this tab.")
