"""Trace-driven episode simulation for evaluating re-planning policies (E5) and for labelling the
learned trigger (S8).

Everything that happens in an episode is real or learned:
  * the backlog at the episode start, and every ADD / DROP / MODIFY / TEAM event with its day, come
    from the replayed Jira history;
  * the planner sees the estimator's predictions (as of each event) and the execution model's
    learned efficiency / capacity forecast;
  * 'truth' is each issue's real logged effort and each developer's real logged hours per sprint.
Only the counterfactual execution of a plan is modelled (validated separately by the E4 back-test).
"""
import copy
import time
from dataclasses import dataclass, field

import numpy as np

from .executor import share
from ..scheduling import ScheduleInstance, Problem, greedy, run_ga, run_pso, run_nsga2, run_mopso

OPTIMISERS = dict(greedy=lambda p, **k: greedy(p), ga=run_ga, pso=run_pso, nsga2=run_nsga2, mopso=run_mopso)


@dataclass
class Task:
    id: str
    req: str
    cluster: int
    weight: float
    due_sprint: float            # absolute sprint index (np.inf = no due date)
    preds: list
    est_hours: float             # planner's current estimate (reference-developer hours)
    true_ref_hours: float        # real effort expressed in reference-developer hours
    done_ref: float = 0.0
    finished_day: float = None
    dropped: bool = False


@dataclass
class EpisodeData:
    """Everything the simulator needs for one project window (built by dynest.replay)."""
    tasks: list                   # Task objects present at day 0
    events: list                  # (day, kind, payload) sorted by day; kinds: ADD, DROP, MODIFY, CAPACITY
    sprint_days: int
    n_sprints: int
    dev_names: list
    eta_plan: np.ndarray          # (D, K) learned efficiency used by the planner
    eta_true: np.ndarray          # (D, K) efficiency used to execute (fitted on the full history of the window)
    cap_true: np.ndarray          # (D, S) real logged hours per developer and sprint
    cap_plan: np.ndarray          # (D, S) planner's capacity forecast (from training data only)
    kappa: float
    handover: float
    wip_limit: np.ndarray = None   # learnt concurrency per developer

    def __post_init__(self):
        if self.wip_limit is None:
            self.wip_limit = np.full(len(self.dev_names), 1)


@dataclass
class EpisodeLog:
    replans: int = 0
    repairs: int = 0
    opt_time: float = 0.0
    churn: list = field(default_factory=list)
    planned_sprints: dict = field(default_factory=dict)       # task -> set of planned sprints
    decisions: list = field(default_factory=list)


class Episode:
    def __init__(self, data, policy, optimiser="ga", weights=None, churn_tol=None, opt_kw=None, seed=0,
                 horizon=6, lock_current=True):
        self.d = data
        self.policy = policy
        self.opt_name, self.opt = optimiser, OPTIMISERS[optimiser]
        self.weights, self.churn_tol = weights, churn_tol
        self.opt_kw = dict(gens=60) if opt_kw is None else opt_kw
        if optimiser == "greedy":
            self.opt_kw = {}
        self.seed, self.horizon, self.lock_current = seed, horizon, lock_current
        self.tasks = {t.id: copy.copy(t) for t in data.tasks}
        self.plan = {}                         # id -> (abs sprint, dev)
        self.day = 0
        self.log = EpisodeLog()
        self.ref_effort = None
        self.D = len(data.dev_names)
        self.worked = np.zeros((self.D, max(data.n_sprints * 3, data.cap_true.shape[1])))
        self.switch_hours = 0.0
        self.wasted_hours = 0.0
        self._seen = {}                        # (dev, sprint) -> set(req)
        self.commit = {}                       # id -> sprint committed at sprint start (carry-over metric)

    # ------------------------------------------------------------------ helpers
    @property
    def sprint(self):
        return self.day // self.d.sprint_days

    def open_tasks(self):
        return [t for t in self.tasks.values() if not t.dropped and t.finished_day is None]

    def scope_effort(self):
        return sum(t.est_hours for t in self.tasks.values() if not t.dropped)

    def _cap(self, mat, s0, H):
        S = mat.shape[1]
        cols = [mat[:, min(s, S - 1)] for s in range(s0, s0 + H)]
        return np.column_stack(cols).astype(float)

    def _instance(self, repair):
        tasks = self.open_tasks()
        ids = [t.id for t in tasks]
        pos = {i: k for k, i in enumerate(ids)}
        s0 = self.sprint
        cap = self._cap(self.d.cap_plan, s0, self.horizon)
        cap[:, 0] *= 1 - (self.day % self.d.sprint_days) / self.d.sprint_days
        prev_s = np.array([self.plan[i][0] - s0 if i in self.plan else -1 for i in ids])
        prev_d = np.array([self.plan[i][1] if i in self.plan else -1 for i in ids])
        prev_s = np.where(prev_d >= 0, np.maximum(prev_s, 0), -1)
        started = np.array([t.done_ref > 0 for t in tasks], dtype=bool)
        if repair:
            locked = prev_d >= 0
        else:
            mid_sprint = self.day % self.d.sprint_days != 0          # sprint planning re-opens the sprint
            locked = (prev_d >= 0) & (started | ((prev_s == 0) & self.lock_current & mid_sprint))
        inst = ScheduleInstance(
            ids=ids, req=[t.req for t in tasks],
            remaining=np.array([max(t.est_hours - t.done_ref, 0.1 * t.est_hours, 0.25) for t in tasks]),
            cluster=np.array([t.cluster for t in tasks]), weight=np.array([t.weight for t in tasks]),
            due=np.array([t.due_sprint - s0 if np.isfinite(t.due_sprint) else 1e9 for t in tasks]),
            preds=[[pos[p] for p in t.preds if p in pos] for t in tasks],
            eta=self.d.eta_plan, capacity=cap, dev_names=self.d.dev_names,
            locked=locked, lock_dev=prev_d, lock_sprint=np.maximum(prev_s, 0),
            assignee=np.where(started, prev_d, -1), done=np.array([t.done_ref for t in tasks]),
            prev_sprint=prev_s, prev_dev=prev_d, kappa=self.d.kappa, handover=self.d.handover)
        return inst

    def _apply(self, inst, sprint_rel, dev, churn=None):
        """Install a plan; churn is measured, not taken from the optimiser: effort-weighted share of
        previously planned open tasks whose sprint (0.5) or developer (0.5) changed."""
        s0 = self.sprint
        new = {}
        for k, i in enumerate(inst.ids):
            new[i] = (int(sprint_rel[k]) + s0, int(dev[k]))
            self.log.planned_sprints.setdefault(i, set()).add(new[i][0])
        both = [k for k, i in enumerate(inst.ids) if i in self.plan]
        if both:
            r = inst.remaining[both]
            mv = np.array([0.5 * (new[inst.ids[k]][0] != max(self.plan[inst.ids[k]][0], s0))
                           + 0.5 * (new[inst.ids[k]][1] != self.plan[inst.ids[k]][1]) for k in both])
            measured = float((r * mv).sum() / max(r.sum(), 1e-9))
        else:
            measured = 0.0
        self.plan = new
        self.log.churn.append(measured)

    def replan(self):
        inst = self._instance(repair=False)
        if inst.n == 0:
            return
        t0 = time.perf_counter()
        prob = Problem(inst, weights=self.weights, seed=self.seed + self.day)
        prob.churn_tol = self.churn_tol
        init = inst.keys_from_plan(inst.prev_sprint, inst.prev_dev, np.random.default_rng(self.seed))
        kw = dict(self.opt_kw)
        if self.opt_name != "greedy":
            kw.update(seed=self.seed + self.day, init=init)
        res = self.opt(prob, **kw)
        plan = inst.decode(res.keys, dev_rule=res.extra.get("dev_rule", "key"))
        self.log.opt_time += time.perf_counter() - t0
        self.log.replans += 1
        self._apply(inst, plan.sprint, plan.dev, float(res.F[3]))
        self.ref_effort = self.scope_effort()

    def repair(self):
        inst = self._instance(repair=True)
        if inst.n == 0 or inst.locked.all():
            return
        plan = inst.decode(inst.greedy_keys(), dev_rule="earliest")
        self.log.repairs += 1
        self._apply(inst, plan.sprint, plan.dev, 0.0)

    # ------------------------------------------------------------------ dynamics
    def _event(self, kind, payload):
        if kind == "ADD":
            t = payload
            self.tasks[t.id] = copy.copy(t)
        elif kind == "DROP":
            t = self.tasks.get(payload)
            if t is not None and t.finished_day is None:
                t.dropped = True
                self.wasted_hours += t.done_ref
                self.plan.pop(t.id, None)
        elif kind == "MODIFY":
            i, est = payload
            if i in self.tasks:
                self.tasks[i].est_hours = est
        elif kind == "CAPACITY":
            pass                                  # capacity truth is already in cap_true; forecasts refresh below

    def _work_day(self):
        """Processor sharing: a developer's daily hours are split over up to wip_limit[d] eligible tasks
        (water-filling), matching how the effort proxy itself was measured."""
        s = self.sprint
        L = self.d.sprint_days
        for d in range(self.D):
            daily = self.d.cap_true[d, min(s, self.d.cap_true.shape[1] - 1)] / L
            budget = daily
            seen = self._seen.setdefault((d, s), set())
            queue = sorted([i for i, (ps, pd) in self.plan.items() if pd == d and ps <= s
                            and self.tasks[i].finished_day is None and not self.tasks[i].dropped],
                           key=lambda i: (self.plan[i][0], i))
            ready = [i for i in queue if not any(p in self.tasks and self.tasks[p].finished_day is None
                                                 and not self.tasks[p].dropped for p in self.tasks[i].preds)]
            active = ready[:max(int(self.d.wip_limit[d]), 1)]
            for i in active:
                t = self.tasks[i]
                if t.req not in seen:
                    if seen:
                        lost = min(self.d.kappa, budget)
                        budget -= lost
                        self.switch_hours += lost
                    seen.add(t.req)
            if budget <= 1e-9 or not active:
                continue
            etas = np.array([max(self.d.eta_true[d, self.tasks[i].cluster], 1e-3) for i in active])
            needs = np.array([(self.tasks[i].true_ref_hours - self.tasks[i].done_ref) for i in active]) / etas
            alloc = share(budget, needs)
            for i, use, eta, need in zip(active, alloc, etas, needs):
                t = self.tasks[i]
                t.done_ref += use * eta
                self.worked[d, s] += use
                if use >= need - 1e-9:
                    t.finished_day = self.day + min(use / max(daily, 1e-9), 1.0)

    def start(self):
        self._k, self._sod = 0, -1
        self.replan()
        return self

    def advance(self, until_day):
        """Simulate until `until_day` (resumable: a cloned episode continues exactly where it was)."""
        L = self.d.sprint_days
        events = self.d.events
        while self.day < until_day:
            if self.day % L == 0 and self._sod != self.day:
                self._sod = self.day
                if self.day > 0 and self.policy.at_sprint_start(self):
                    self.replan()
                for i, (ps, _) in self.plan.items():
                    if ps == self.sprint and i not in self.commit:
                        self.commit[i] = ps
            while self._k < len(events) and events[self._k][0] <= self.day:
                _, kind, payload = events[self._k]
                self._event(kind, payload)
                self._k += 1
                decision = self.policy.decide(self, kind)
                self.log.decisions.append((self.day, kind, decision))
                if decision == "replan":
                    self.replan()
                else:
                    self.repair()
            self._work_day()
            self.day += 1
            if self._k >= len(events) and not self.open_tasks():
                break
        return self

    def run(self, max_days=None):
        max_days = max_days or self.d.sprint_days * self.d.n_sprints * 3
        return self.start().advance(max_days).outcome()

    def outcome(self):
        L = self.d.sprint_days
        done = [t for t in self.tasks.values() if not t.dropped]
        fin = np.array([(t.finished_day if t.finished_day is not None else self.day + L) / L for t in done])
        w = np.array([t.weight for t in done])
        due = np.array([t.due_sprint for t in done])
        late = np.where(np.isfinite(due), np.maximum(0, fin - (due + 1)), 0)
        carry = [t.finished_day is None or t.finished_day / L > self.commit[t.id] + 1
                 for t in done if t.id in self.commit]
        nerv = [len(v) for k, v in self.log.planned_sprints.items()]
        used = int(np.ceil(fin.max())) if len(fin) else 0
        dl = self.worked[:, :max(used, 1)].sum(1)
        return dict(
            completed=float(np.mean([t.finished_day is not None for t in done])) if done else 1.0,
            makespan_sprints=float(fin.max()) if len(fin) else 0.0,
            realised_hours=float(self.worked.sum() + self.switch_hours),
            wasted_hours=float(self.wasted_hours),
            switch_hours=float(self.switch_hours),
            weighted_completion=float((w * fin).sum() / w.sum()) if len(w) else 0.0,
            weighted_tardiness=float((w * late).sum()),
            sla_miss_rate=float((late > 0)[np.isfinite(due)].mean()) if np.isfinite(due).any() else 0.0,
            carry_over_rate=float(np.mean(carry)) if carry else 0.0,
            leveling=float(dl.mean() / dl.max()) if dl.max() > 0 else 1.0,
            replans=self.log.replans, repairs=self.log.repairs,
            churn_total=float(np.sum(self.log.churn)), nervousness=float(np.mean(nerv)) if nerv else 1.0,
            opt_time_s=float(self.log.opt_time))

    def clone(self):
        """Copy the mutable simulation state; the read-only EpisodeData and policy are shared."""
        return copy.deepcopy(self, memo={id(self.d): self.d, id(self.policy): self.policy})
