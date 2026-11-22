"""Command-line demo on real data: replay a held-out TAWOS project window through DynEst-R.

  python run_demo.py                 # first run trains the origin-0 model (cached afterwards)
  python run_demo.py --project 1     # pick a project id (see experiments/results/s0_audit.csv)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
from common import dataset, efficiency_models, fitted_model, origin_splits, replay_builder  # noqa: E402
from dynest.sim import Threshold  # noqa: E402
from dynest.tracker import ReplayTracker  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--project", type=int, default=None)
ap.add_argument("--backend", default="minilm", choices=["minilm", "tfidf"])
ap.add_argument("--tau", type=float, default=0.05)
a = ap.parse_args()
pd.set_option("display.width", 200, "display.max_columns", 30)

ds = dataset()
f, tr, va, te = origin_splits()[0]
print(f"Training / loading DynEst-R on {len(tr):,} real requirements from {len(ds.projects)} projects ...")
m = fitted_model(0, tr, va, a.backend)
eff_plan, eff_true = efficiency_models(m, tr, va, te)
print("Execution model:", {k: round(v, 3) for k, v in eff_plan.summary().items()})

p = a.project or int(te.project_id.value_counts().index[0])
rb = replay_builder()
w = te[te.project_id == p]
T0 = rb.episode_starts(p, w.resolved_at.min(), w.resolved_at.max())[0]
ep, info = rb.build(p, T0, 4, m, eff_plan, eff_true)
print(f"\nProject {ds.keys.get(p, p)}: window from {T0.date()}, {len(ep.tasks)} open requirements, "
      f"{len(ep.events)} real change events, {len(ep.dev_names)} developers, sprint = {ep.sprint_days} working days")

trk = ReplayTracker(ep, m, Threshold(a.tau)).run()
h = trk.history_frame()
print("\nChange log (first 25 events):")
print(h.head(25).round(1).to_string(index=False))
print("\nOutcome:", {k: round(v, 2) for k, v in trk.outcome().items()})

row = info["fr"].iloc[0]
feats = rb.ctx.features_as_of(pd.Series([info["entry"].iloc[0]], index=[info["fr"].index[0]]))
ex = m.explain(feats)
print(f"\nWhy this estimate? issue {row.issue_key} | cluster '{ex['cluster']}' | ANFIS complexity {ex['c_hat']:.2f}")
for r in ex.get("rules", []):
    print(f"   {r['firing_strength']:.2f}  {r['rule']}  (support {r['support']})")
