"""Dynamic requirement tracker (S8) for the dashboard / demo: step a real replayed episode event by event.

Each snapshot records the change event, the re-estimated scope effort with its conformal project
interval, the effort/cost deltas, the cumulative drift since the last optimisation, the policy
decision (re-optimise vs repair) and the plan (with churn) after the decision.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sim import Episode, Learned, Threshold


@dataclass
class Snapshot:
    version: int
    day: int
    event: str
    effort: float
    low: float
    high: float
    delta: float
    pct_delta: float
    drift: float
    decision: str
    plan: dict = field(default_factory=dict)
    churn: float = 0.0


class _StepPolicy:
    """Wraps a policy and records a snapshot after every decision."""

    def __init__(self, inner, tracker):
        self.inner, self.tracker = inner, tracker
        self.name = inner.name

    def at_sprint_start(self, ep):
        return self.inner.at_sprint_start(ep)

    def decide(self, ep, kind):
        ref = ep.ref_effort or ep.scope_effort()
        drift = (ep.scope_effort() - ref) / max(ref, 1e-9)
        d = self.inner.decide(ep, kind)
        self.tracker._pending = (kind, drift, d)
        return d


class ReplayTracker:
    def __init__(self, ep_data, model, policy=None, optimiser="ga", rate=None, opt_kw=None, weights=None, seed=0):
        self.model, self.rate = model, rate
        self.policy = policy or Threshold(0.05)
        self.ep = Episode(ep_data, _StepPolicy(self.policy, self), optimiser, weights=weights,
                          opt_kw=opt_kw or dict(gens=40, pop=40), seed=seed)
        self.snapshots = []
        self._pending = None

    def _snap(self, event, decision, drift):
        E = self.ep.scope_effort()
        n = sum(1 for t in self.ep.tasks.values() if not t.dropped)
        lo, hi = self.model.project_interval(E, n) if self.model is not None else (E, E)
        prev = self.snapshots[-1].effort if self.snapshots else E
        self.snapshots.append(Snapshot(len(self.snapshots), self.ep.day, event, E, lo, hi, E - prev,
                                       (E - prev) / prev if prev else 0.0, drift, decision, dict(self.ep.plan),
                                       self.ep.log.churn[-1] if self.ep.log.churn else 0.0))

    def run(self):
        """Replay the whole episode; one snapshot per change event (plus the initial plan)."""
        ep = self.ep
        ep.start()
        self._snap("INITIAL plan", "replan", 0.0)
        L = ep.d.sprint_days
        max_days = L * ep.d.n_sprints * 3
        n_dec = 0
        while ep.day < max_days and (ep._k < len(ep.d.events) or ep.open_tasks()):
            ep.advance(ep.day + 1)
            new = ep.log.decisions[n_dec:]
            for day, kind, d in new:
                payload = ep.d.events[n_dec][2]
                label = f"{kind} {payload.id if kind == 'ADD' else payload[0] if kind == 'MODIFY' else payload}"
                _, drift, _ = self._pending if self._pending else (kind, 0.0, d)
                self._snap(label, d, drift)
                n_dec += 1
            if ep._k >= len(ep.d.events) and not ep.open_tasks():
                break
        return self

    def history_frame(self):
        return pd.DataFrame([dict(version=s.version, day=s.day, event=s.event, effort=s.effort, low=s.low, high=s.high,
                                  delta=s.delta, pct_delta=100 * s.pct_delta, drift=100 * s.drift, decision=s.decision,
                                  churn=s.churn) for s in self.snapshots])

    def outcome(self):
        return self.ep.outcome()
