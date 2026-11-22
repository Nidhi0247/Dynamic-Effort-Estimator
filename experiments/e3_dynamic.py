"""E3 tracking (RQ3), E4 execution-model back-test, E5 re-planning policies + tau sensitivity (RQ4).

Episodes are real project windows (4 sprints from a real sprint start).  Validation-window episodes
train the learned trigger and tune tau; test-window episodes are used for evaluation only.
Common random numbers: every policy replays exactly the same events with the same truth.
Unit of analysis = episode; Friedman + Wilcoxon/Holm + A12; bootstrap CIs over episodes.
"""
from common import args, efficiency_models, episodes, fitted_model, log, origin_splits, replay_builder, save
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from dynest import stats
from dynest.replay import backtest
from dynest.sim import Episode, Hybrid, Learned, SprintBoundary, Static, Threshold, trigger_features

TAUS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
PRIMARY = ["realised_hours", "weighted_completion", "weighted_tardiness", "churn_total"]


# ---------------------------------------------------------------------------------------- E3
def tracking(ep):
    """Scope-effort tracking error after every event: dynamic re-estimation vs static estimates."""
    est = {t.id: t.est_hours for t in ep.tasks}
    true = {t.id: t.true_ref_hours for t in ep.tasks}
    static_ids = set(est)
    static0 = dict(est)
    added_est = {}
    dropped = set()
    rows = []
    for day, kind, payload in ep.events:
        if kind == "ADD":
            est[payload.id] = payload.est_hours
            true[payload.id] = payload.true_ref_hours
            added_est[payload.id] = payload.est_hours
        elif kind == "DROP":
            dropped.add(payload)
        elif kind == "MODIFY":
            i, h = payload
            if i in est:
                est[i] = h
        live = [i for i in est if i not in dropped]
        E_true = sum(true[i] for i in live)
        if E_true <= 0:
            continue
        E_dyn = sum(est[i] for i in live)
        E_static = sum(static0[i] for i in static_ids if i not in dropped)
        E_static_add = E_static + sum(added_est[i] for i in live if i in added_est)
        rows.append(dict(day=day, dynamic=abs(E_dyn - E_true) / E_true, static=abs(E_static - E_true) / E_true,
                         static_plus_adds=abs(E_static_add - E_true) / E_true))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- trigger
def _J(o, w_sum):
    return o["weighted_completion"] + o["weighted_tardiness"] / max(w_sum, 1e-9)


class _Labeller(Static):
    """At sampled events, branch immediately (re-optimise now vs repair only), run both 2 sprints ahead,
    keep only the features and the net benefit (no episode copies are retained)."""

    def __init__(self, every, rng):
        self.every, self.rng = every, rng
        self.X, self.gain = [], []

    def decide(self, ep, kind):
        if self.rng.random() < self.every:
            until = ep.day + 2 * ep.d.sprint_days
            wsum = sum(t.weight for t in ep.tasks.values() if not t.dropped)
            out = []
            for act in ("replan", "repair"):
                c = ep.clone()
                c.policy = Static()
                getattr(c, act)()
                out.append(c.advance(until).outcome())
            oa, ob = out
            self.X.append(trigger_features(ep, kind))
            self.gain.append(_J(ob, wsum) - _J(oa, wsum) - oa["churn_total"] + ob["churn_total"])
        return "repair"


def label_episode(ep_data, seed, opt_kw, sample=0.25):
    """Label = net benefit of re-optimising now over repair-only, 2 sprints ahead."""
    lab = _Labeller(sample, np.random.default_rng(seed))
    Episode(ep_data, lab, "ga", opt_kw=opt_kw, seed=seed).run()
    return lab.X, [g > 0 for g in lab.gain], lab.gain


def train_trigger(val_eps, opt_kw, jobs):
    res = Parallel(n_jobs=jobs)(delayed(label_episode)(e, i, opt_kw) for i, (e, _) in enumerate(val_eps))
    X = np.array([x for r in res for x in r[0]])
    y = np.array([v for r in res for v in r[1]])
    g = np.array([v for r in res for v in r[2]])
    if len(X) < 10 or len(set(y)) < 2:
        log("  trigger: not enough labelled decisions; learned policy falls back to tau tuned on validation")
        return None, dict(n=len(X))
    clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    thr = max(np.unique(np.round(p, 3)), key=lambda t: g[p >= t].sum())          # max net benefit
    log(f"  trigger: {len(X)} labelled decisions, {y.mean():.0%} positive, threshold {thr:.2f}")
    return Learned(clf, thr), dict(n=len(X), positive=float(y.mean()), threshold=float(thr),
                                   coef=dict(zip(["abs_drift", "signed_drift", "unscheduled_share", "days_left_share",
                                                  "open_tasks_log", "is_add", "is_drop", "is_modify"],
                                                 np.round(clf.coef_[0], 3))))


# ---------------------------------------------------------------------------------------- E5
def run_policy(ep_data, policy, optimiser, opt_kw, seed, weights=None, churn_tol=None):
    out = Episode(ep_data, policy, optimiser, weights=weights, churn_tol=churn_tol, opt_kw=opt_kw, seed=seed).run()
    return dict(policy=policy.name if optimiser != "greedy" else f"{policy.name} [greedy]", **out)


def main():
    a = args([(["--per-project"], dict(type=int, default=2)), (["--gens"], dict(type=int, default=40))])
    opt_kw = dict(gens=10 if a.smoke else a.gens, pop=20 if a.smoke else 40)
    rb = replay_builder()
    track, bt, pol_rows, trig_info = [], [], [], []
    for k, (f, tr, va, te) in enumerate(origin_splits(a.smoke)):
        m = fitted_model(k, tr, va, a.backend, a.smoke)
        eff_plan, eff_true = efficiency_models(m, tr, va, te)
        log(f"origin {k}: efficiency {({x: round(v, 3) for x, v in eff_plan.summary().items()})}")
        val_eps = episodes(rb, m, eff_plan, eff_true, va, per_project=1, smoke=a.smoke)
        test_eps = episodes(rb, m, eff_plan, eff_true, te, per_project=a.per_project, smoke=a.smoke)
        log(f"  episodes: validation={len(val_eps)} test={len(test_eps)}")
        # E3 + E4
        for j, (ep, info) in enumerate(test_eps):
            t = tracking(ep)
            if len(t):
                track.append(dict(origin=k, episode=j, project=info["project"], n_events=len(t),
                                  **t[["dynamic", "static", "static_plus_adds"]].mean().to_dict()))
            bt.append(dict(origin=k, episode=j, project=info["project"], **backtest(ep, info)))
        # tau tuned on validation episodes (composite J), learned trigger trained on validation episodes
        jobs = [(e, Threshold(t)) for e, _ in val_eps for t in TAUS]
        vres = Parallel(n_jobs=a.jobs)(delayed(run_policy)(e, p, "ga", opt_kw, 0) for e, p in jobs)
        v = pd.DataFrame(vres)
        v["J"] = v.weighted_completion + v.churn_total
        tau_star = float(v.groupby("policy").J.mean().idxmin().split("=")[1])
        learned, info = train_trigger(val_eps, opt_kw, a.jobs)
        trig_info.append(dict(origin=k, tau_star=tau_star, **info))
        log(f"  tuned tau* = {tau_star}")
        policies = [Static(), SprintBoundary(), Hybrid(0.05)] + [Threshold(t) for t in TAUS]
        tuned = Threshold(tau_star)
        tuned.name = "tuned-tau"
        policies.append(tuned)
        if learned is not None:
            policies.append(learned)
        jobs = [(j, e, p, "ga") for j, (e, _) in enumerate(test_eps) for p in policies]
        jobs += [(j, e, Threshold(0.05), "greedy") for j, (e, _) in enumerate(test_eps)]
        res = Parallel(n_jobs=a.jobs)(delayed(run_policy)(e, p, o, opt_kw, j) for j, e, p, o in jobs)
        for (j, e, p, o), r in zip(jobs, res):
            pol_rows.append(dict(origin=k, episode=j, project=test_eps[j][1]["project"], **r))
        log(f"  origin {k}: {len(res)} policy runs done")
    T, B, P = pd.DataFrame(track), pd.DataFrame(bt), pd.DataFrame(pol_rows)
    save(T, "e3_tracking.csv")
    save(B, "e4_backtest.csv")
    save(P, "e5_policies.csv")
    save(pd.DataFrame(trig_info), "e5_trigger.csv")
    report(T, B, P)


def report(T, B, P):
    if len(T):
        print("\nE3 tracking error (mean relative error of scope effort, per episode):")
        print(T[["dynamic", "static", "static_plus_adds"]].describe().round(3).to_string())
        out = stats.compare(T[["dynamic", "static", "static_plus_adds"]])
        print(out["pairwise"][["a", "b", "median_diff", "ci_low", "ci_high", "p_holm", "A12", "magnitude"]].round(3).to_string(index=False))
    if len(B):
        print("\nE4 back-test of the execution model (team's real assignment):")
        print(B[["mae_sprints", "same_sprint", "rank_corr"]].describe().round(3).to_string())
    if len(P):
        P["unit"] = P.origin.astype(str) + "-" + P.episode.astype(str)
        cols = ["realised_hours", "weighted_completion", "weighted_tardiness", "sla_miss_rate", "carry_over_rate",
                "churn_total", "nervousness", "replans", "opt_time_s", "completed"]
        print("\nE5 policy outcomes (median over episodes):")
        print(P.groupby("policy")[cols].median().round(3).to_string())
        main_pols = [p for p in P.policy.unique() if not p.startswith("tau=") or p in ("tau=0", "tau=0.05", "tau=0.05 [greedy]")]
        for metric in PRIMARY:
            t = P[P.policy.isin(main_pols)].pivot_table(index="unit", columns="policy", values=metric)
            out = stats.compare(t)
            print(f"\n[{metric}] Friedman p={out.get('friedman_p', float('nan')):.3g}; mean ranks:",
                  {k: round(v, 2) for k, v in sorted(out["mean_rank"].items(), key=lambda x: x[1])})
            save(out["pairwise"], f"e5_pairwise_{metric}.csv")
        tau = P[P.policy.str.startswith("tau=") & ~P.policy.str.contains("greedy")].copy()
        tau["tau"] = tau.policy.str.split("=").str[1].astype(float)
        rows = []
        for t, g in tau.groupby("tau"):
            r = dict(tau=t)
            for mtr in PRIMARY + ["replans"]:
                lo, hi = stats.bootstrap_ci(g[mtr].values)
                r.update({mtr: float(g[mtr].median()), f"{mtr}_lo": lo, f"{mtr}_hi": hi})
            rows.append(r)
        S = pd.DataFrame(rows)
        save(S, "e5_tau_sensitivity.csv")
        print("\nTau sensitivity (median [95% BCa CI]):")
        print(S[["tau", "replans", "churn_total", "weighted_completion", "weighted_tardiness", "realised_hours"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
