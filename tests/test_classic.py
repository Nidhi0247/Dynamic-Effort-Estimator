"""Classic benchmark (E8): loaders, COCOMO multiplier table, estimator protocol."""
import numpy as np
import pytest

from dynest.classic.datasets import RAW, em_table, load_all
from dynest.classic.models import Estimator, NeuroFuzzyCore, Prep, estimators, ablations

pytestmark = pytest.mark.skipif(not (RAW / "coc81_1_1.arff").exists(), reason="run experiments/get_classic_data.py")

SIZES = dict(Albrecht=24, Kemerer=15, Desharnais=77, China=499, Maxwell=63, Kitchenham=145, Miyazaki94=48,
             COCOMO81=63, NASA93=93)


def test_loaders_sizes_and_no_post_hoc_columns():
    D = load_all()
    assert {k: d.n for k, d in D.items()} == SIZES
    for d in D.values():
        assert (d.y > 0).all() and d.size in d.X
        assert not any("duration" in c.lower() or "estimate" in c.lower() for c in d.X.columns)
        assert not d.X.isna().any().any()


def test_em_table_matches_boehm_1981():
    t = em_table()
    assert t["cplx"] == {"vl": 0.70, "l": 0.85, "n": 1.0, "h": 1.15, "vh": 1.30, "xh": 1.65}
    assert t["acap"]["vh"] == 0.71 and t["pcap"]["vl"] == 1.42 and t["sced"]["vl"] == 1.23


def test_prep_fitted_on_training_categories_only():
    d = load_all(["Maxwell"])["Maxwell"]
    p = Prep(d.cats).fit(d.X.iloc[:30])
    Z = p.transform(d.X)
    assert Z.shape == (d.n, len(p.names)) and np.isfinite(Z).all()


def test_estimators_fit_predict_and_intervals():
    d = load_all(["Albrecht"])["Albrecht"]
    tr, te = np.arange(4, d.n), np.arange(4)
    for e in estimators(d):
        e.fit(d.X.iloc[tr], d.y[tr], d, n_trials=2, inner=3, seed=0)
        p, lo, hi = e.predict(d.X.iloc[te])
        assert np.isfinite(p).all() and (lo <= p).all() and (p <= hi).all(), e.name


def test_neurofuzzy_rules_and_ablations():
    d = load_all(["COCOMO81"])["COCOMO81"]
    e = Estimator("NF", NeuroFuzzyCore, fixed=dict(__import__("dynest.classic.models", fromlist=["x"]).NF_DEFAULT))
    e.fit(d.X, d.y, d, inner=3)
    assert len(e.model.fis.rules) > 0 and len(e.model.fis.rule_table()) == len(e.model.fis.rules)
    for a in ablations(e.hp):
        a.fit(d.X.iloc[10:], d.y[10:], d, inner=3)
        assert np.isfinite(a.predict(d.X.iloc[:10])[0]).all(), a.name
