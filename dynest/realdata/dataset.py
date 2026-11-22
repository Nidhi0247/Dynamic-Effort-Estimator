"""Study dataset: gate projects, modelling rows, components, and leakage-free splits.

Splits are made per project in resolution-time order and pooled (each project contributes its own
past to training and its own future to validation/test), so a model never sees a project's future.
A one-sprint gap separates each project's training cutoff from its validation window: training rows
must be resolved at least `gap` before the first validation resolution.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .frame import build_frame, modelling_rows
from .store import Store

AUDIT = Path(__file__).resolve().parents[2] / "experiments" / "results" / "s0_audit.csv"
MIN_ROWS, MIN_SPRINTS = 150, 10


class Dataset:
    def __init__(self, store=None, log=print):
        self.store = store or Store()
        audit = pd.read_csv(AUDIT)
        self.audit = audit
        pids = sorted(audit[audit.sprints >= MIN_SPRINTS].project_id)
        self.frame = build_frame(pids, self.store, log=log)
        rows = modelling_rows(self.frame)
        counts = rows.groupby("project_id").size()
        sprints = audit.set_index("project_id").sprints
        self.projects = sorted(p for p, n in counts.items() if n >= MIN_ROWS and sprints.get(p, 0) >= MIN_SPRINTS)
        self.rows = rows[rows.project_id.isin(self.projects)].sort_values("resolved_at").reset_index(drop=True)
        self.keys = audit.set_index("project_id").key.to_dict()
        comp = self.store.components
        self.components = comp.groupby("Issue_ID").Name.apply(list).to_dict()

    def splits(self, origins=(0.55, 0.65, 0.75), val=0.1, test=0.1, gap=pd.Timedelta(days=14), projects=None):
        """Yield (origin, train, val, test) DataFrames pooled over projects."""
        rows = self.rows if projects is None else self.rows[self.rows.project_id.isin(projects)]
        for f in origins:
            parts = {"train": [], "val": [], "test": []}
            for _, g in rows.groupby("project_id"):
                g = g.sort_values("resolved_at")
                n = len(g)
                a, b, c = int(f * n), int((f + val) * n), int((f + val + test) * n)
                va, te = g.iloc[a:b], g.iloc[b:c]
                if len(va) == 0 or len(te) == 0:
                    continue
                tr = g.iloc[:a]
                tr = tr[tr.resolved_at <= va.resolved_at.min() - gap]
                parts["train"].append(tr)
                parts["val"].append(va)
                parts["test"].append(te)
            yield f, *(pd.concat(parts[k]).reset_index(drop=True) for k in ("train", "val", "test"))

    def lopo(self, val_frac=0.15):
        """Leave-one-project-out: train/val on all other projects (time split), test = held-out project."""
        for p in self.projects:
            other = self.rows[self.rows.project_id != p].sort_values("resolved_at")
            cut = int(len(other) * (1 - val_frac))
            yield p, other.iloc[:cut].reset_index(drop=True), other.iloc[cut:].reset_index(drop=True), \
                self.rows[self.rows.project_id == p].reset_index(drop=True)
