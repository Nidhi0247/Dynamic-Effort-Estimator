"""Real-data tests: leakage guarantees, effort proxy, splits, pipeline and replay (need data/tawos)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "tawos" / "Issue.parquet"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="converted TAWOS data not present (see README)")


@pytest.fixture(scope="module")
def ds():
    from dynest.realdata.dataset import Dataset
    return Dataset(log=lambda *a: None)


def test_effort_proxy_wip_sharing():
    from dynest.realdata.effort import wip_adjusted_hours, weekday_hours
    t = pd.Timestamp("2024-01-08 09:00")                       # a Monday
    iv = pd.DataFrame(dict(start=[t, t], end=[t + pd.Timedelta(hours=24)] * 2, assignee=[1.0, 1.0]))
    h = wip_adjusted_hours(iv)
    assert np.allclose(h, 4.0)                                 # 8 business hours shared by two tasks
    sat = pd.Timestamp("2024-01-13")
    assert weekday_hours(np.array([sat], "datetime64[s]"), np.array([sat + pd.Timedelta(days=1)], "datetime64[s]"))[0] == 0


def test_decision_time_precedes_work(ds):
    r = ds.rows
    assert (r.t_decision < r.work_start).all()
    assert (r.t_decision >= r.created_at).all()


def test_as_of_values_ignore_later_changes(ds):
    """SP and description as of t must equal the value before any later changelog entry."""
    from dynest.realdata.frame import context
    ctx = context(sorted(ds.audit[ds.audit.sprints >= 10].project_id), log=lambda *a: None)
    ch = ctx.byf["Story Points"]
    ch = ch[ch.Issue_ID.isin(ds.rows.issue_id)].sort_values("Creation_Date")
    sample = ch.groupby("Issue_ID").head(1).head(50)
    T = pd.Series((sample.Creation_Date - pd.Timedelta(seconds=1)).values, index=sample.Issue_ID.values)
    f = ctx.features_as_of(T).set_index("issue_id")
    before = pd.to_numeric(sample.set_index("Issue_ID").From_String, errors="coerce")
    ok = before.notna()
    assert np.allclose(f.sp[before.index[ok]].astype(float), before[ok].astype(float))


def test_splits_are_time_ordered_per_project(ds):
    f, tr, va, te = next(ds.splits(origins=(0.55,)))
    for p in ds.projects:
        a, b, c = tr[tr.project_id == p], va[va.project_id == p], te[te.project_id == p]
        if len(a) and len(b):
            assert a.resolved_at.max() < b.resolved_at.min()
        if len(b) and len(c):
            assert b.resolved_at.max() <= c.resolved_at.min()
    assert not set(tr.issue_id) & set(te.issue_id)


def test_pipeline_and_replay_small(ds):
    from dynest import DynEstR
    from dynest.realdata.frame import context
    from dynest.replay import ReplayBuilder, backtest, fit_efficiency
    from dynest.sim import Episode, Threshold
    p = 1
    f, tr, va, te = next(ds.splits(origins=(0.55,), projects=[p]))
    m = DynEstR(dict(text_backend="tfidf", mamdani_gens=3, anfis_epochs=10, som_epochs=5, som_grid=3)).fit(tr, va, ds.components)
    pr = m.predict(te)
    assert (pr.hours > 0).all() and (pr.low <= pr.hours).all() and (pr.hours <= pr.high).all()
    assert len(m.fis.rules) > 0
    eff = fit_efficiency(m, tr)
    rb = ReplayBuilder(context(sorted(ds.audit[ds.audit.sprints >= 10].project_id), log=lambda *a: None), ds.frame)
    for start in rb.episode_starts(p, te.resolved_at.min(), te.resolved_at.max()):
        built = rb.build(p, start, 2, m, eff, eff)
        if built:
            break
    ep, info = built
    assert all(e[0] >= 0 for e in ep.events)
    assert backtest(ep, info)["n"] >= 0
    out = Episode(ep, Threshold(0.05), "ga", opt_kw=dict(gens=5, pop=10)).run()
    assert out["replans"] >= 1 and 0 <= out["completed"] <= 1
