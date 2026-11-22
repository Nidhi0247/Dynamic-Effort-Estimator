"""Shared experiment harness: dataset, per-origin trained models (cached), replay episodes."""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynest import DynEstR, tune  # noqa: E402
from dynest.realdata.dataset import Dataset  # noqa: E402
from dynest.realdata.frame import context  # noqa: E402
from dynest.replay import ReplayBuilder, fit_efficiency  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
MODELS = ROOT / "data" / "cache" / "models"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

ORIGINS = (0.55, 0.65, 0.75)


def args(extra=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny configuration for a quick end-to-end check")
    ap.add_argument("--backend", default="minilm", choices=["minilm", "tfidf"])
    ap.add_argument("--jobs", type=int, default=8)
    for a in extra or []:
        ap.add_argument(*a[0], **a[1])
    return ap.parse_args()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


_DS = None


def dataset():
    global _DS
    if _DS is None:
        _DS = Dataset(log=lambda *a: None)
    return _DS


def origin_splits(smoke=False):
    ds = dataset()
    origins = ORIGINS[:1] if smoke else ORIGINS
    return list(ds.splits(origins=origins))


def fitted_model(k, train, val, backend="minilm", smoke=False, cfg=None, tag="full"):
    """DynEst-R tuned on the validation window of origin k (cached on disk)."""
    name = f"dynestr_o{k}_{backend}_{tag}{'_smoke' if smoke else ''}.pkl"
    path = MODELS / name
    if path.exists():
        return pickle.loads(path.read_bytes())
    base = {**(cfg or {}), "text_backend": backend}
    if smoke:
        base.update(mamdani_gens=5, anfis_epochs=15, som_epochs=8)
    ds = dataset()
    if tag == "full":
        m = tune(train, val, ds.components, n_trials=2 if smoke else 8, base_cfg=base, seed=k, log=log)
    else:
        m = DynEstR(base).fit(train, val, ds.components)
    path.write_bytes(pickle.dumps(m))
    return m


def replay_builder():
    ds = dataset()
    ctx = context(sorted(ds.audit[ds.audit.sprints >= 10].project_id), log=lambda *a: None)
    return ReplayBuilder(ctx, ds.frame)


def episodes(rb, model, eff_plan, eff_true, window, n_sprints=4, per_project=2, smoke=False, projects=None):
    """Episodes starting at real sprint starts inside each project's window (DataFrame of rows)."""
    out = []
    projs = projects or sorted(window.project_id.unique())
    if smoke:
        projs = projs[:3]
        per_project = 1
    for p in projs:
        w = window[window.project_id == p]
        if len(w) == 0:
            continue
        starts = rb.episode_starts(p, w.resolved_at.min(), w.resolved_at.max())
        if not starts:
            continue
        pick = [starts[int(i)] for i in np.linspace(0, len(starts) - 1, min(per_project, len(starts)))]
        for T0 in dict.fromkeys(pick):
            try:
                built = rb.build(p, T0, n_sprints, model, eff_plan, eff_true)
            except Exception as e:                      # a malformed project window should not stop the study
                log(f"  skip episode p={p} T0={T0}: {e}")
                continue
            if built is not None and len(built[0].tasks) + len(built[0].events) >= 5:
                out.append(built)
    return out


def efficiency_models(model, train, val, test):
    return fit_efficiency(model, train), fit_efficiency(model, pd.concat([train, val, test]))


def save(df, name):
    path = RESULTS / name
    df.to_csv(path, index=False)
    log(f"wrote {path.relative_to(ROOT)}")
    return path
