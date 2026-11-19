"""Command-line demo: baseline estimate, then a stream of requirement changes (Steps 1-8)."""
import pandas as pd
from dynest import EffortPipeline, RequirementTracker, demo_project, demo_team, benchmark

pd.set_option("display.width", 200, "display.max_columns", 30)
print("Training SOM + ANFIS + Mamdani calibration on historical database ...")
pipe = EffortPipeline().fit()
print(pipe.metrics.round(3), "\n")

reqs = demo_project()
tracker = RequirementTracker(pipe, demo_team(), threshold=0.05, opt_iters=100)


def show(snap, label):
    e = snap.estimate
    print(f"\n=== v{snap.version}: {label}")
    for ev in snap.events:
        print("   event:", ev)
    print(f"   effort {e.effort:6.1f} ±{e.band:4.1f} person-days | duration {e.duration_days:5.1f} d "
          f"(team {e.team_size}) | Δeffort {snap.effort_delta:+.1f} ({100 * snap.pct_delta:+.1f}%) | "
          f"cost ₹{snap.cost:,.0f} (Δ ₹{snap.cost_delta:+,.0f})")
    if snap.optimization:
        for name, r in snap.optimization["results"].items():
            print(f"   {name:3s}: fitness {r.fitness:.3f} | converged @iter {r.conv_iter:3d} | {r.time_s:.2f}s | "
                  f"leveling {r.metrics['leveling_efficiency']:.2f} | util {r.metrics['utilization']:.2f} | "
                  f"moved {r.metrics['tasks_moved']} tasks")
    else:
        print("   (below threshold -> optimiser not triggered)" + ("  [plan stale]" if snap.plan_stale else ""))


show(tracker.update(reqs), "initial baseline")

reqs = pd.concat([reqs, pd.DataFrame([dict(id="R13", title="Real-time chat support", ui=6, backend=8, data=3,
        complexity=8, integration=7, clarity=3, experience=4, size_points=8, priority=2, active=True)])], ignore_index=True)
show(tracker.update(reqs), "client adds a vague, complex requirement")

reqs.loc[reqs.id == "R10", "active"] = False
show(tracker.update(reqs), "client drops customer-report export")

reqs.loc[reqs.id == "R03", ["clarity", "integration"]] = [3, 10]
show(tracker.update(reqs), "payment provider changes -> R03 becomes vague and hard to integrate")

reqs.loc[reqs.id == "R02", "size_points"] = 6
show(tracker.update(reqs), "minor tweak to landing pages")

print("\nFinal sprint plan (best of GA/PSO):")
snap = [s for s in tracker.snapshots if s.optimization][-1]
best = snap.optimization["results"][snap.optimization["winner"]]
plan = snap.optimization["problem"].plan(best.genes).sort_values(["sprint", "developer"])
print(plan.to_string(index=False))

print("\nGA vs PSO benchmark (5 seeds) on the current backlog:")
summ, _, _ = benchmark(snap.optimization["problem"], runs=5)
print(summ.T)
