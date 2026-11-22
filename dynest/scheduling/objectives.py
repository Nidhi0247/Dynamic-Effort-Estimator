"""Four minimised objectives of a decoded plan, plus per-instance normalisation and scalarisation.

f1 cost      = sum_i c_{i,d_i} + kappa * sum_{d,s} max(0, n_ds - 1)                    (hours)
f2 schedule  = sum_i w_i C_i / sum w_i + sum_i w_i max(0, C_i - (due_i + 1)) / sum w_i  (sprints)
f3 workload  = sum_{d,s} max(0, load_ds - C_ds) / sum C_ds  +  CV_d(u_d),  u_d = sum_s load / sum_s C
f4 churn     = sum_{i known} r_i (0.5 [s_i != s_i^prev] + 0.5 [d_i != d_i^prev]) / sum_{i known} r_i
"""
import numpy as np

OBJECTIVES = ["cost", "schedule", "workload", "churn"]


def objectives(inst, plan):
    i = np.arange(inst.n)
    c = inst.cost[i, plan.dev]
    nreq = plan.n_requirements()
    f1 = c.sum() + inst.kappa * np.maximum(nreq - 1, 0).sum()
    w = inst.weight
    f2 = (w * plan.completion).sum() / w.sum() + (w * np.maximum(0, plan.completion - (inst.due + 1))).sum() / w.sum()
    over = np.maximum(plan.load - plan.cap, 0).sum() / max(plan.cap.sum(), 1e-9)
    act = inst.active_dev
    u = plan.load[act].sum(1) / np.maximum(plan.cap[act].sum(1), 1e-9)
    cv = u.std() / (u.mean() + 1e-9) if len(u) > 1 else 0.0
    f3 = over + cv
    known = inst.prev_sprint >= 0
    if known.any():
        r = inst.remaining[known]
        mv = 0.5 * (plan.sprint[known] != inst.prev_sprint[known]) + 0.5 * (plan.dev[known] != inst.prev_dev[known])
        f4 = float((r * mv).sum() / max(r.sum(), 1e-9))
    else:
        f4 = 0.0
    return np.array([f1, f2, f3, f4], float)


def indicators(inst, plan):
    """Interpretable plan indicators for reporting."""
    w = inst.weight
    load, cap = plan.load, plan.cap
    act = inst.active_dev
    dl = load[act].sum(1)
    known = inst.prev_sprint >= 0
    moved = int(((plan.sprint != inst.prev_sprint) | (plan.dev != inst.prev_dev))[known].sum()) if known.any() else 0
    late = plan.completion > inst.due + 1
    return dict(makespan=float(plan.completion.max()), overload_h=float(np.maximum(load - cap, 0).sum()),
                leveling=float(dl.mean() / dl.max()) if dl.max() > 0 else 1.0,
                utilization=float(load.sum() / max(cap.sum(), 1e-9)),
                weighted_tardiness=float((w * np.maximum(0, plan.completion - (inst.due + 1))).sum()),
                sla_miss_rate=float(late[inst.due < 1e6].mean()) if (inst.due < 1e6).any() else 0.0,
                tasks_moved=moved, sprints=int(plan.horizon))


class Normalizer:
    """f~_k = (f_k - z*_k) / (zN_k - z*_k); ideal/nadir learnt per instance from sampled plans."""

    def fit(self, F):
        F = np.asarray(F, float)
        nd = F[nondominated(F)]
        self.ideal = F.min(0)
        self.nadir = nd.max(0)
        span = self.nadir - self.ideal
        self.span = np.where(span < 1e-9, 1.0, span)
        return self

    def __call__(self, F):
        # not clipped below 0: optimisers routinely beat the probed ideal and must stay comparable
        return np.minimum((np.asarray(F, float) - self.ideal) / self.span, 2.0)


def nondominated(F):
    F = np.asarray(F, float)
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        if keep[i]:
            dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
            if dom.any():
                keep[i] = False
    return keep


def scalarize(Fn, w):
    return np.asarray(Fn, float) @ np.asarray(w, float)


def tchebycheff(Fn, w, rho=0.01):
    Fn = np.atleast_2d(Fn)
    w = np.asarray(w, float)
    return (w * Fn).max(1) + rho * Fn.sum(1)
