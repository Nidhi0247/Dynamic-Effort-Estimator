"""Correctness tests for the method modules (toy inputs only check the maths; no model is trained on them)."""
import numpy as np
import pytest
from scipy import stats as sps

from dynest.anfis import ANFIS
from dynest.conformal import ConformalIntervals
from dynest.learned_mamdani import LearnedMamdani, partition_memberships
from dynest.scheduling import (ScheduleInstance, Problem, run_ga, run_pso, run_nsga2, run_mopso, greedy,
                               hypervolume, fit_preferences)
from dynest.sim import execute
from dynest import stats


def test_partition_is_strong():
    x = np.linspace(-1, 11, 500)
    mu = partition_memberships(x, [0, 3, 7, 10])
    assert np.allclose(mu.sum(1), 1.0)
    assert (mu >= 0).all() and (mu <= 1).all()


def test_wang_mendel_recovers_monotone_rules():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (600, 2))
    y = X[:, 0] + 0.2 * X[:, 1]
    fis = LearnedMamdani(["a", "b"], n_terms=(3, 3), n_out=5).fit(X, y, tune=False)
    lo = fis.predict(np.array([[0.05, 0.5]]))[0]
    hi = fis.predict(np.array([[0.95, 0.5]]))[0]
    assert hi > lo + 0.5
    assert fis.coverage(X) == 1.0
    _, fired, _ = fis.explain([0.9, 0.9])
    assert fired and "THEN effort IS" in fired[0]["rule"]


def test_mamdani_tuning_does_not_hurt_holdout():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 1, (500, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    a = LearnedMamdani(["a", "b"], (4, 4), 7, seed=0).fit(X, y, tune=False)
    b = LearnedMamdani(["a", "b"], (4, 4), 7, seed=0).fit(X, y, tune=True, gens=10, pop=12)
    assert b.tune_history[-1] <= b.tune_history[0] + 1e-12
    ca = b.in_centres[0]
    assert np.all(np.diff(ca) > 0)                                    # order preserved by repair


def test_anfis_rls_matches_batch():
    rng = np.random.default_rng(2)
    X = rng.uniform(0, 1, (300, 2))
    t = X[:, 0] * X[:, 1] + 0.1 * rng.normal(size=300)
    m = ANFIS(2, 2).fit(X[:200], t[:200], epochs=5)
    m.partial_fit(X[200:], t[200:])
    A = m._design(X)[0]
    batch = np.linalg.solve(A.T @ A + 1e-6 * np.eye(A.shape[1]), A.T @ t)
    assert np.allclose(m.theta, batch, atol=1e-4)


def test_conformal_coverage():
    rng = np.random.default_rng(3)
    F = rng.uniform(0, 1, (4000, 2))
    r = rng.normal(0, 0.2 + 0.5 * F[:, 0])
    ci = ConformalIntervals(alpha=0.1).fit(F[:2000], r[:2000])
    lo, hi, _ = ci.interval(np.ones(2000), F[2000:])
    cov = ((np.exp(r[2000:]) >= lo) & (np.exp(r[2000:]) <= hi)).mean()
    assert cov >= 0.87


def _instance(seed=0, n=24, D=3, H=5):
    rng = np.random.default_rng(seed)
    preds = [[j for j in range(i) if rng.random() < 0.06] for i in range(n)]
    locked = np.zeros(n, bool)
    locked[:2] = True
    return ScheduleInstance(ids=[f"T{i}" for i in range(n)], req=[f"R{i // 2}" for i in range(n)],
                            remaining=rng.uniform(4, 30, n), cluster=rng.integers(0, 2, n),
                            weight=rng.integers(1, 4, n), due=rng.integers(1, 5, n).astype(float), preds=preds,
                            eta=rng.uniform(0.7, 1.3, (D, 2)), capacity=np.full((D, H), 50.0),
                            dev_names=["a", "b", "c"], locked=locked, lock_dev=np.r_[0, 1, [-1] * (n - 2)],
                            kappa=1.0, handover=0.2)


def test_decoder_constraints_hold():
    inst = _instance()
    rng = np.random.default_rng(0)
    for _ in range(500):
        p = inst.decode(rng.random(2 * inst.n))
        for i, ps in enumerate(inst.preds):
            for j in ps:
                assert p.sprint[i] >= p.sprint[j] + (0 if p.dev[i] == p.dev[j] else 1)
        assert (p.sprint[:2] == 0).all() and list(p.dev[:2]) == [0, 1]
        assert (p.load <= p.cap + 1e-6).all()


def test_optimisers_improve_on_random():
    inst = _instance(1)
    prob = Problem(inst, seed=0)
    rng = np.random.default_rng(5)
    rand = np.median(prob.score(np.array([prob.raw(rng.random(2 * inst.n)) for _ in range(50)])))
    for f in (run_ga, run_pso):
        r = f(prob, gens=25)
        assert r.score < rand and r.history[-1] <= r.history[0]
    for f in (run_nsga2, run_mopso):
        r = f(prob, gens=15)
        assert len(r.front_F) >= 1


def test_hypervolume_known_values():
    assert np.isclose(hypervolume(np.array([[0.0, 0.0]]), ref=1.0), 1.0)
    assert np.isclose(hypervolume(np.array([[0.5, 0.0], [0.0, 0.5]]), ref=1.0), 0.75)
    assert np.isclose(hypervolume(np.array([[0.5, 0.5, 0.5]]), ref=1.0), 0.125)


def test_preference_learning_recovers_weights():
    rng = np.random.default_rng(0)
    w_true = np.array([0.6, 0.2, 0.1, 0.1])
    pairs = []
    for _ in range(80):
        alts = rng.random((6, 4))
        ch = alts[np.argmin(alts @ w_true)]
        pairs.append((ch, np.delete(alts, np.argmin(alts @ w_true), 0)))
    res = fit_preferences(pairs)
    assert np.argmax(res["weights"]) == 0 and res["pairwise_accuracy"] > 0.85


def test_executor_conserves_work():
    inst = _instance(2)
    p = inst.decode(inst.greedy_keys(), dev_rule="earliest")
    hours = inst.cost[np.arange(inst.n), p.dev]
    out = execute(inst, p.sprint, p.dev, hours, inst.capacity)
    assert out["completed_all"]
    assert np.isclose(out["realised_hours"], hours.sum() + out["switch_hours"])
    for i, ps in enumerate(inst.preds):
        for j in ps:
            assert out["completion"][i] >= out["completion"][j] - 1e-9


def test_stats_against_scipy():
    rng = np.random.default_rng(0)
    x, y = rng.normal(0, 1, 40), rng.normal(0.5, 1, 40)
    assert np.isclose(stats.a12(x, y), 1 - stats.a12(y, x))
    adj = stats.holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    import pandas as pd
    t = pd.DataFrame(dict(a=x, b=y, c=x + 0.1))
    out = stats.compare(t)
    assert np.isclose(out["friedman_p"], sps.friedmanchisquare(x, y, x + 0.1).pvalue)
