"""Plan executor: plays a plan forward against 'true' task hours and capacity in discrete time steps.

Used (a) to back-test the execution model on the team's real historical sprints (E4) and
(b) to score counterfactual plans produced by each re-planning policy (E5).

Each developer works its queue (planned sprint, planned completion order); a task can start once
its predecessors are finished; starting a new requirement inside a sprint costs kappa hours.
Unfinished work carries over.  Returns realised completion times and outcome indicators.
"""
import numpy as np


def share(budget, needs):
    """Water-filling: split `budget` equally over tasks, re-sharing what finished tasks do not need."""
    needs = np.asarray(needs, float)
    alloc = np.zeros(len(needs))
    left = list(range(len(needs)))
    b = budget
    while left and b > 1e-12:
        each = b / len(left)
        done = [k for k in left if needs[k] - alloc[k] <= each]
        if not done:
            for k in left:
                alloc[k] += each
            break
        for k in done:
            b -= needs[k] - alloc[k]
            alloc[k] = needs[k]
        left = [k for k in left if k not in done]
    return alloc


def execute(inst, sprint, dev, true_hours, capacity, steps=10, max_sprints=None, kappa=None, wip_limit=None):
    """
    inst        ScheduleInstance (preds, weight, due, req)
    sprint, dev planned assignment (relative sprints)
    true_hours  (n,) hours the assigned developer really needs for each task
    capacity    (D, H) hours really available (extended with its last column when needed)
    wip_limit   (D,) tasks a developer works on concurrently (processor sharing); default 1
    """
    n, D = inst.n, inst.D
    kappa = inst.kappa if kappa is None else kappa
    W = np.ones(D, int) if wip_limit is None else np.maximum(np.asarray(wip_limit, int), 1)
    H0 = capacity.shape[1]
    max_sprints = max_sprints or max(3 * H0, int(np.max(sprint)) + 6)
    cap = np.concatenate([capacity] + [capacity[:, -1:]] * max(0, max_sprints - H0), axis=1)
    left = np.asarray(true_hours, float).copy()
    finish = np.full(n, np.inf)
    queues = [sorted(np.where(dev == d)[0], key=lambda i: (sprint[i], i)) for d in range(D)]
    worked = np.zeros((D, max_sprints))
    switch_h = 0.0
    for s in range(max_sprints):
        seen = [set() for _ in range(D)]
        for k in range(steps):
            now = s + k / steps
            for d in range(D):
                budget = cap[d, s] / steps
                ready = [i for i in queues[d] if finish[i] == np.inf and sprint[i] <= s
                         and all(finish[j] <= now for j in inst.preds[i])][:W[d]]
                if not ready or budget <= 1e-12:
                    continue
                for i in ready:
                    if inst.req[i] not in seen[d]:
                        if seen[d]:
                            left[i] += kappa
                            switch_h += kappa
                        seen[d].add(inst.req[i])
                alloc = share(budget, left[ready])
                for i, a in zip(ready, alloc):
                    left[i] -= a
                    worked[d, s] += a
                    if left[i] <= 1e-9:
                        finish[i] = s + (k + min(a / budget, 1.0)) / steps
                queues[d] = [i for i in queues[d] if finish[i] == np.inf]
        if np.isfinite(finish).all():
            break
    done = np.isfinite(finish)
    fin = np.where(done, finish, max_sprints + 1.0)
    w = inst.weight
    late = np.maximum(0, fin - (inst.due + 1))
    planned_end = np.asarray(sprint, float) + 1
    used = int(np.ceil(fin.max()))
    act = cap[:, :used].sum(1) > 0
    dl = worked[:, :used].sum(1)[act]
    return dict(
        completion=fin, completed_all=bool(done.all()),
        makespan=float(fin.max()),
        realised_hours=float(np.asarray(true_hours).sum() + switch_h),
        switch_hours=float(switch_h),
        weighted_completion=float((w * fin).sum() / w.sum()),
        weighted_tardiness=float((w * late).sum()),
        sla_miss_rate=float((late > 0)[inst.due < 1e6].mean()) if (inst.due < 1e6).any() else 0.0,
        carry_over_rate=float((fin > planned_end + 1e-9).mean()),
        leveling=float(dl.mean() / dl.max()) if len(dl) and dl.max() > 0 else 1.0,
        idle_share=float(1 - worked[:, :used].sum() / max(cap[:, :used].sum(), 1e-9)),
    )
