"""Re-planning policies (S8).  decide() is called after every real change event; at_sprint_start()
at each sprint boundary.  'repair' = insert new/changed tasks without moving anything else."""
import numpy as np


class Static:
    name = "static"

    def decide(self, ep, kind):
        return "repair"

    def at_sprint_start(self, ep):
        return False


class SprintBoundary(Static):
    name = "sprint-boundary"

    def at_sprint_start(self, ep):
        return True


class Threshold(Static):
    """Re-optimise when cumulative estimated-scope drift since the last optimisation reaches tau."""

    def __init__(self, tau):
        self.tau = tau
        self.name = f"tau={tau:g}"

    def decide(self, ep, kind):
        ref = ep.ref_effort or ep.scope_effort()
        drift = abs(ep.scope_effort() - ref) / max(ref, 1e-9)
        return "replan" if drift >= self.tau - 1e-12 else "repair"


class Hybrid(Threshold):
    def __init__(self, tau):
        super().__init__(tau)
        self.name = f"hybrid(tau={tau:g})"

    def at_sprint_start(self, ep):
        return True


TRIGGER_FEATURES = ["abs_drift", "signed_drift", "unscheduled_share", "days_left_share", "open_tasks_log",
                    "is_add", "is_drop", "is_modify"]


def trigger_features(ep, kind):
    ref = ep.ref_effort or ep.scope_effort()
    sd = (ep.scope_effort() - ref) / max(ref, 1e-9)
    open_ = ep.open_tasks()
    est_open = sum(t.est_hours for t in open_) or 1.0
    unsched = sum(t.est_hours for t in open_ if t.id not in ep.plan) / est_open
    L = ep.d.sprint_days
    return np.array([abs(sd), sd, unsched, 1 - (ep.day % L) / L, np.log1p(len(open_)),
                     kind == "ADD", kind == "DROP", kind == "MODIFY"], float)


class Learned(Static):
    """Logistic trigger trained on replay labels (see dynest.replay.label_trigger)."""
    name = "learned"

    def __init__(self, model, threshold=0.5):
        self.model, self.threshold = model, threshold

    def decide(self, ep, kind):
        p = self.model.predict_proba(trigger_features(ep, kind)[None])[0, 1]
        return "replan" if p >= self.threshold else "repair"
