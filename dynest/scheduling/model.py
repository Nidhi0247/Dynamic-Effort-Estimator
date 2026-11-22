"""Dynamic sprint scheduling instance (S9) and the random-key serial schedule-generation decoder.

All parameters are supplied by trained / measured components:
  remaining effort r_i (estimator + logged progress), cluster k_i (SOM), developer efficiency
  eta[d, k] and capacity C[d, s] (execution model), context-switch kappa and handover h
  (execution model), dependencies (issue links), locks (current sprint commitment).

Hard constraints guaranteed by the decoder:
  H1 precedence: s_i >= s_j + 1 if d_i != d_j, s_i >= s_j if d_i == d_j, for each edge j -> i
  H2 locks: locked tasks keep their sprint (default: current sprint 0) and developer
  H3 nothing is scheduled in the past (sprint 0 = current sprint, remaining capacity only)
  H4 capacity: unlocked work never exceeds C[d, s] (horizon extends instead)
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScheduleInstance:
    ids: list                    # task ids (requirement id or id#chunk)
    req: list                    # requirement id per task (for context switching)
    remaining: np.ndarray        # r_i, hours
    cluster: np.ndarray          # k_i
    weight: np.ndarray           # w_i > 0 (priority)
    due: np.ndarray              # due sprint index (relative; large = none)
    preds: list                  # list of predecessor index lists
    eta: np.ndarray              # (D, K) efficiency (hours of reference work per hour)
    capacity: np.ndarray         # (D, H) hours available per developer-sprint
    dev_names: list
    locked: np.ndarray = None    # bool (n,)
    lock_dev: np.ndarray = None  # int (n,)
    lock_sprint: np.ndarray = None   # relative sprint for locked tasks (default 0 = current sprint)
    assignee: np.ndarray = None  # current assignee index (-1 none)
    done: np.ndarray = None      # effort already spent (hours) – for handover cost
    prev_sprint: np.ndarray = None   # previous plan (relative sprint), -1 = new task
    prev_dev: np.ndarray = None
    kappa: float = 0.0           # context-switch overhead per extra requirement in a dev-sprint (hours)
    handover: float = 0.0        # extra share of done work when an in-progress task changes developer
    max_horizon_factor: int = 3
    _cost: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        n, D = len(self.ids), self.eta.shape[0]
        self.n, self.D = n, D
        z = lambda v, f: np.asarray(v if v is not None else np.full(n, f))
        self.locked = z(self.locked, False).astype(bool)
        self.lock_dev = z(self.lock_dev, -1).astype(int)
        self.lock_sprint = z(self.lock_sprint, 0).astype(int)
        self.assignee = z(self.assignee, -1).astype(int)
        self.done = z(self.done, 0.0).astype(float)
        self.prev_sprint = z(self.prev_sprint, -1).astype(int)
        self.prev_dev = z(self.prev_dev, -1).astype(int)
        self.remaining = np.asarray(self.remaining, float)
        self.cluster = np.asarray(self.cluster, int)
        self.weight = np.asarray(self.weight, float)
        self.due = np.asarray(self.due, float)
        eff = np.maximum(self.eta[:, self.cluster].T, 1e-3)                  # (n, D)
        cost = self.remaining[:, None] / eff
        moved = (self.assignee[:, None] >= 0) & (np.arange(D)[None] != self.assignee[:, None])
        cost += moved * self.handover * self.done[:, None]
        self._cost = cost
        self.active_dev = self.capacity.sum(1) > 0
        self._order_dev = np.argsort(cost, axis=1, kind="stable")              # cheapest first
        self._topo_ok()

    @property
    def cost(self):
        return self._cost

    def _topo_ok(self):
        indeg = np.array([len(p) for p in self.preds])
        succ = [[] for _ in range(self.n)]
        for i, ps in enumerate(self.preds):
            for j in ps:
                succ[j].append(i)
        self.succ = succ
        # break cycles (bad link data) by dropping back-edges found by Kahn's algorithm
        ready = [i for i in range(self.n) if indeg[i] == 0]
        seen, deg = 0, indeg.copy()
        while ready:
            i = ready.pop()
            seen += 1
            for s in succ[i]:
                deg[s] -= 1
                if deg[s] == 0:
                    ready.append(s)
        if seen < self.n:
            cyc = set(np.where(deg > 0)[0])
            self.preds = [[j for j in ps if not (i in cyc and j in cyc)] for i, ps in enumerate(self.preds)]
            self._topo_ok()

    # ------------------------------------------------------------------ decoding
    def decode(self, z, dev_rule="key"):
        """z in [0,1]^(2n): priority keys then developer keys.  Returns a Plan."""
        n, D = self.n, self.D
        zp, zd = np.asarray(z[:n]), np.asarray(z[n:])
        H0 = self.capacity.shape[1]
        Hmax = H0 * self.max_horizon_factor
        cap = np.concatenate([self.capacity] + [self.capacity[:, -1:]] * (Hmax - H0), axis=1)
        load = np.zeros((D, Hmax))
        reqs = [[set() for _ in range(Hmax)] for _ in range(D)]
        sprint = np.full(n, -1)
        dev = np.full(n, -1)
        compl = np.zeros(n)
        cost = self._cost

        def place(i, d, s):
            c = cost[i, d]
            extra = self.kappa if (reqs[d][s] and self.req[i] not in reqs[d][s]) else 0.0
            load[d, s] += c + extra
            reqs[d][s].add(self.req[i])
            sprint[i], dev[i] = s, d
            compl[i] = s + min(load[d, s] / max(cap[d, s], 1e-9), 1.0)

        for i in np.where(self.locked)[0]:
            d = self.lock_dev[i] if self.lock_dev[i] >= 0 else int(self._order_dev[i, 0])
            place(i, d, min(max(self.lock_sprint[i], 0), Hmax - 1))

        indeg = np.array([sum(1 for j in ps if not self.locked[j]) for ps in self.preds])
        indeg[self.locked] = -1
        ready = [i for i in range(n) if indeg[i] == 0]
        eligible = [d for d in range(D) if self.active_dev[d]] or list(range(D))
        while ready:
            i = min(ready, key=lambda t: zp[t])
            ready.remove(i)
            order = [d for d in self._order_dev[i] if d in eligible]
            if dev_rule == "key":
                cands = [order[min(int(zd[i] * len(order)), len(order) - 1)]]
            else:                                   # "earliest": try every developer, earliest completion
                cands = order
            best = None
            for d in cands:
                es = 0
                for j in self.preds[i]:
                    if sprint[j] >= 0:
                        es = max(es, sprint[j] + (0 if dev[j] == d else 1))
                c = cost[i, d]
                s_ok = None
                for s in range(es, Hmax):
                    extra = self.kappa if (reqs[d][s] and self.req[i] not in reqs[d][s]) else 0.0
                    if load[d, s] + c + extra <= cap[d, s] + 1e-9 or (load[d, s] == 0 and cap[d, s] > 0 and c > cap[d, s]):
                        s_ok = s
                        break
                if s_ok is None:
                    s_ok = Hmax - 1
                fin = s_ok + (load[d, s_ok] + c) / max(cap[d, s_ok], 1e-9)
                if best is None or fin < best[0]:
                    best = (fin, d, s_ok)
            place(i, best[1], best[2])
            for s_ in self.succ[i]:
                if indeg[s_] > 0:
                    indeg[s_] -= 1
                    if indeg[s_] == 0:
                        ready.append(s_)
        H = int(max(sprint.max() + 1, 1))
        return Plan(sprint, dev, compl, load[:, :H], cap[:, :H], reqs, H)

    # ------------------------------------------------------------------ helpers
    def keys_from_plan(self, sprint, dev, rng):
        """Encode an existing plan (e.g. the previous one) as random keys for warm starts."""
        n = self.n
        zp, zd = rng.random(n), rng.random(n)
        known = sprint >= 0
        if known.any():
            order = np.lexsort((np.arange(n), sprint))
            ranks = np.empty(n)
            ranks[order] = np.arange(n)
            zp[known] = (ranks[known] + 0.5) / n
        for i in np.where(dev >= 0)[0]:
            order = [d for d in self._order_dev[i] if self.active_dev[d]]
            if dev[i] in order:
                zd[i] = (order.index(dev[i]) + 0.5) / len(order)
        return np.r_[zp, zd]

    def greedy_keys(self):
        """Priority-first, due-date-first list order (reference heuristic, used with dev_rule='earliest')."""
        n = self.n
        order = np.lexsort((-self.remaining, -self.weight, self.due))
        zp = np.empty(n)
        zp[order] = (np.arange(n) + 0.5) / n
        return np.r_[zp, np.zeros(n)]


@dataclass
class Plan:
    sprint: np.ndarray
    dev: np.ndarray
    completion: np.ndarray       # fractional completion sprint
    load: np.ndarray             # (D, H)
    cap: np.ndarray              # (D, H)
    reqs: list
    horizon: int

    def n_requirements(self):
        return np.array([[len(self.reqs[d][s]) for s in range(self.horizon)] for d in range(len(self.reqs))])
