import time
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .data import ATTRS
from .optimizers import SprintProblem, run_both

WATCH = ATTRS + ["priority"]
@dataclass
class Snapshot:
    version: int
    timestamp: float
    events: list
    estimate: object
    effort_delta: float = 0.0
    pct_delta: float = 0.0
    duration_delta: float = 0.0
    cost: float = 0.0
    cost_delta: float = 0.0
    cum_pct: float = 0.0           
    triggered: bool = False
    plan_stale: bool = False
    optimization: dict = field(default_factory=dict)   


class RequirementTracker:
    def __init__(self, pipeline, team, focus=0.8, rate=4000.0, threshold=0.05, sprint_days=10,
                 auto_optimize=True, opt_iters=100, seed=0):
        self.pipe, self.team = pipeline, team
        self.focus, self.rate, self.threshold, self.sprint_days = focus, rate, threshold, sprint_days
        self.auto_optimize, self.opt_iters, self.seed = auto_optimize, opt_iters, seed
        self.current = None
        self.config = None
        self.snapshots: list[Snapshot] = []
        self.plan = {}                
        self.last_problem = None
        self.last_results = None
        self._opt_effort = None         

    @staticmethod
    def diff(old, new):
        ev = []
        o = old.set_index("id") if old is not None else pd.DataFrame(columns=new.columns).set_index("id")
        n = new.set_index("id")
        for i in n.index:
            now = bool(n.loc[i, "active"])
            if i not in o.index:
                if now:
                    ev.append(f"ADDED {i} ({n.loc[i, 'title']})")
                continue
            was = bool(o.loc[i, "active"])
            if now and not was:
                ev.append(f"ADDED {i} ({n.loc[i, 'title']})")
            elif was and not now:
                ev.append(f"DROPPED {i} ({n.loc[i, 'title']})")
            elif now and was:
                ch = [c for c in WATCH if not np.isclose(float(o.loc[i, c]), float(n.loc[i, c]))]
                if ch:
                    ev.append(f"MODIFIED {i}: " + ", ".join(f"{c} {o.loc[i, c]:g}→{n.loc[i, c]:g}" for c in ch))
        for i in o.index.difference(n.index):
            if bool(o.loc[i, "active"]):
                ev.append(f"DROPPED {i} ({o.loc[i, 'title']})")
        return ev

    def update(self, reqs, force=False):
        reqs = reqs.copy().reset_index(drop=True)
        reqs["active"] = reqs["active"].astype(bool)
        cfg = (tuple(self.team["name"]), tuple(self.team["specialty"]), tuple(self.team["speed"]),
               self.focus, self.rate)
        events = self.diff(self.current, reqs) if self.current is not None else ["INITIAL baseline"]
        if self.config is not None and cfg != self.config:
            events.append("CONFIG team / focus / rate changed")
        if not events and not force:
            return None
        
        self.current, self.config = reqs, cfg
        est = self.pipe.estimate_project(reqs, len(self.team), self.focus)
        prev = self.snapshots[-1] if self.snapshots else None
        snap = Snapshot(len(self.snapshots), time.time(), events or ["manual re-run"], est, cost=est.effort * self.rate)
        if prev:
            snap.effort_delta = est.effort - prev.estimate.effort
            snap.pct_delta = snap.effort_delta / prev.estimate.effort if prev.estimate.effort else 0.0
            snap.duration_delta = est.duration_days - prev.estimate.duration_days
            snap.cost_delta = snap.cost - prev.cost
        ref = self._opt_effort
        snap.cum_pct = (est.effort - ref) / ref if ref else 0.0
        snap.triggered = prev is None or ref is None or abs(snap.cum_pct) >= self.threshold or force
        if snap.triggered:
            self._opt_effort = est.effort            # new reference point for drift measurement
        if snap.triggered and self.auto_optimize and len(est.table):
            self.optimize(snap)
        elif len(est.table):
            ids = set(est.table["id"])
            snap.plan_stale = ids != set(self.plan)
        self.snapshots.append(snap)
        return snap

    def optimize(self, snap=None):
        snap = snap or self.snapshots[-1]
        t = snap.estimate.table
        prob = SprintProblem.from_estimates(t, self.team, sprint_days=self.sprint_days, focus=self.focus,
                                            prev=self.plan)
        res = run_both(prob, seed=self.seed + snap.version, iters=self.opt_iters)
        best = min(res.values(), key=lambda r: r.fitness)
        self.plan = prob.plan_map(best.genes)
        self.last_problem, self.last_results = prob, res
        snap.optimization = dict(results=res, problem=prob, winner=best.name)
        snap.plan_stale = False
        self._opt_effort = snap.estimate.effort
        return snap

    def history_frame(self):
        return pd.DataFrame([dict(version=s.version, events="; ".join(s.events), effort=s.estimate.effort,
                                  low=s.estimate.low, high=s.estimate.high, duration_days=s.estimate.duration_days,
                                  cost=s.cost, effort_delta=s.effort_delta, pct_delta=100 * s.pct_delta,
                                  cost_delta=s.cost_delta, reoptimised=bool(s.optimization)) for s in self.snapshots])
