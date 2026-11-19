import numpy as np
import pytest
from dynest import (EffortPipeline, RequirementTracker, SprintProblem, run_ga, run_pso,
                    demo_project, demo_team, make_history)
from dynest.mamdani import MamdaniFIS
from dynest.optimizers import greedy_plan


@pytest.fixture(scope="module")
def pipe():
    return EffortPipeline(seed=1).fit(make_history(600, seed=3))


def test_mamdani_monotone_and_bounded():
    f = MamdaniFIS()
    lo = f.infer_batch(1, 9, 9)[0][0]
    hi = f.infer_batch(9, 1, 1)[0][0]
    assert 0 <= lo < hi <= 3
    assert f.infer_batch(5, 2, 5)[0][0] > f.infer_batch(5, 9, 5)[0][0]      # vaguer -> more effort


def test_mamdani_explain_returns_rules():
    _, fired, agg = MamdaniFIS().explain(8, 2, 3)
    assert fired and all(0 < r["firing_strength"] <= 1 for r in fired) and agg.max() > 0


def test_som_labels_cover_clusters(pipe):
    assert set(pipe.som.neuron_labels()) == {0, 1, 2}
    assert pipe.som.qe_history[-1] < pipe.som.qe_history[0]


def test_anfis_learns(pipe):
    assert pipe.anfis_rmse_test < 1.0          # rating on 0-10 scale


def test_pipeline_accuracy(pipe):
    assert pipe.metrics.loc["SOM+ANFIS+Mamdani (proposed)", "MMRE"] < 0.30


def test_effort_responds_to_requirements(pipe):
    d = demo_project()
    base = pipe.estimate_project(d, 4).effort
    d2 = d.copy(); d2.loc[0, "active"] = False
    assert pipe.estimate_project(d2, 4).effort < base
    d3 = d.copy(); d3.loc[0, "clarity"] = 1
    assert pipe.estimate_project(d3, 4).effort > base
    e = pipe.estimate_project(d, 4)
    assert e.low < e.effort < e.high


def test_tracker_events_and_triggers(pipe):
    d = demo_project()
    tr = RequirementTracker(pipe, demo_team(), auto_optimize=False)
    assert tr.update(d).triggered
    assert tr.update(d) is None                                   # nothing changed
    d2 = d.copy(); d2.loc[5, "active"] = False                    # drop the biggest requirement
    s = tr.update(d2)
    assert s.triggered and s.effort_delta < 0 and s.events[0].startswith("DROPPED")
    d3 = d2.copy(); d3.loc[1, "size_points"] = 6                  # tiny tweak
    s = tr.update(d3)
    assert s.events[0].startswith("MODIFIED") and not s.triggered


def test_tracker_runs_optimizers(pipe):
    tr = RequirementTracker(pipe, demo_team(), opt_iters=30)
    s = tr.update(demo_project())
    assert set(s.optimization["results"]) == {"GA", "PSO"} and tr.plan


def test_optimizers_beat_random_and_are_feasible(pipe):
    t = pipe.estimate_project(demo_project(), 4).table
    p = SprintProblem.from_estimates(t, demo_team())
    rnd = np.mean([p.evaluate(np.random.default_rng(i).integers(0, p.ub)) for i in range(50)])
    ga, pso = run_ga(p, iters=60), run_pso(p, iters=60)
    for r in (ga, pso):
        assert r.fitness < rnd and r.metrics["overload_days"] < 3
        assert r.history[-1] <= r.history[0]
    assert ga.fitness <= p.evaluate(greedy_plan(p)) * 1.05
