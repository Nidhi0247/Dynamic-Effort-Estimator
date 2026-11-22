"""E7 preference learning (weights from the team's real plans) and E6 optimiser comparison (RQ5).

E7: for each validation-window episode the team's real plan (developer = real assignee, sprint = sprint
    in which work really started) is compared with feasible alternative plans of the same instance; weights
    are fitted by pairwise logistic likelihood and evaluated on test-window episodes against equal weights.
E6: GA, PSO (scalarised with the learnt weights), NSGA-II, MOPSO and greedy on real re-planning instances
    (episode start + after the first sprint, with the previous plan for churn); R seeds each; HV / IGD+ in
    each instance's normalised space; Dirichlet robustness of the GA-vs-PSO verdict around the learnt weights.
"""
from common import args, efficiency_models, episodes, fitted_model, log, origin_splits, replay_builder, save
import dataclasses

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dynest import stats
from dynest.scheduling import (Problem, fit_preferences, greedy, hypervolume, igd_plus, nondominated, objectives,
                               pairwise_accuracy, run_ga, run_mopso, run_nsga2, run_pso)
from dynest.sim import Episode, Static


def instances(ep_data):
    e = Episode(ep_data, Static())
    first = e._instance(repair=False)
    out = [("start", first)] if first.n >= 3 else []
    e.start()
    e.advance(ep_data.sprint_days)
    inst = e._instance(repair=False)
    if inst.n >= 3:
        out.append(("after-sprint-1", inst))
    return out


def assignment_F(inst, sprint, dev):
    fixed = dataclasses.replace(inst, locked=np.ones(inst.n, bool), lock_sprint=np.asarray(sprint),
                                lock_dev=np.asarray(dev))
    return objectives(fixed, fixed.decode(np.zeros(2 * inst.n)))


def real_plan_pair(ep_data, info, gens):
    """(F_chosen, F_alternatives) in the instance's normalised space for the team's real day-0 plan."""
    inst = Episode(ep_data, Static())._instance(repair=False)
    if inst.n < 3:
        return None
    fr, T0, L = info["fr"], info["T0"], info["L_cal"]
    di = {str(d): k for k, d in enumerate(info["devs"])}
    dev = np.array([di.get(str(int(fr.Assignee_ID[int(i)])), 0) if pd.notna(fr.Assignee_ID[int(i)]) else 0
                    for i in inst.ids])
    ws = fr.work_start.reindex([int(i) for i in inst.ids])
    sprint = np.array([max(int((w - T0).days // L), 0) if pd.notna(w) else 0 for w in ws])
    # alternatives = feasible plans the team could have made (random keys through the same decoder);
    # Pareto-optimal alternatives would dominate any real plan and make every weight vector fail
    prob = Problem(inst, seed=0)
    rng = np.random.default_rng(0)
    alts = np.array([prob.raw(rng.random(2 * inst.n)) for _ in range(40)])
    return prob.norm(assignment_F(inst, sprint, dev)), prob.norm(alts)


def run_algos(inst, w, seeds, gens, churn_tol):
    prob = Problem(inst, weights=w, seed=0)
    prob.churn_tol = churn_tol
    init = inst.keys_from_plan(inst.prev_sprint, inst.prev_dev, np.random.default_rng(0)) if (inst.prev_dev >= 0).any() else None
    rows, fronts = [], {}
    g = greedy(prob)
    rows.append(dict(algo="Greedy", seed=0, score=g.score, time_s=g.time_s, conv_iter=0, **dict(zip("f1 f2 f3 f4".split(), g.F))))
    for s in range(seeds):
        for name, fn in (("GA", run_ga), ("PSO", run_pso), ("NSGA-II", run_nsga2), ("MOPSO", run_mopso)):
            r = fn(prob, gens=gens, seed=s, init=init)
            sc = float(prob.score(r.F[None])[0])
            rows.append(dict(algo=name, seed=s, score=sc, time_s=r.time_s, conv_iter=r.conv_iter,
                             **dict(zip("f1 f2 f3 f4".split(), r.F))))
            Fset = r.front_F if r.front_F is not None else r.F[None]
            fronts[(name, s)] = prob.norm(Fset)
    ref = np.vstack(list(fronts.values()))
    ref = ref[nondominated(ref)]
    for row in rows:
        key = (row["algo"], row["seed"])
        if key in fronts:
            row["HV"] = hypervolume(fronts[key])
            row["IGDplus"] = igd_plus(fronts[key], ref)
    return rows


def dirichlet_verdict(inst, w, n, seeds, gens, rng):
    out = []
    for k in range(n):
        wk = rng.dirichlet(20 * np.asarray(w) + 1)
        prob = Problem(inst, weights=wk, seed=0)
        ga = np.mean([run_ga(prob, gens=gens, seed=s).score for s in range(seeds)])
        pso = np.mean([run_pso(prob, gens=gens, seed=s).score for s in range(seeds)])
        out.append(dict(sample=k, ga=ga, pso=pso, ga_better=ga < pso, **{f"w{j}": v for j, v in enumerate(wk)}))
    return out


def main():
    a = args([(["--seeds"], dict(type=int, default=10)), (["--gens"], dict(type=int, default=80))])
    seeds, gens = (2, 15) if a.smoke else (a.seeds, a.gens)
    rb = replay_builder()
    pref_rows, algo_rows, dir_rows = [], [], []
    for k, (f, tr, va, te) in enumerate(origin_splits(a.smoke)):
        m = fitted_model(k, tr, va, a.backend, a.smoke)
        eff_plan, eff_true = efficiency_models(m, tr, va, te)
        val_eps = episodes(rb, m, eff_plan, eff_true, va, per_project=2, smoke=a.smoke)
        test_eps = episodes(rb, m, eff_plan, eff_true, te, per_project=2, smoke=a.smoke)
        # ---------------- E7
        tr_pairs = [p for p in Parallel(n_jobs=a.jobs)(delayed(real_plan_pair)(e, i, gens) for e, i in val_eps) if p]
        te_pairs = [p for p in Parallel(n_jobs=a.jobs)(delayed(real_plan_pair)(e, i, gens) for e, i in test_eps) if p]
        pref = fit_preferences(tr_pairs) if tr_pairs else dict(weights=np.full(4, 0.25), pairwise_accuracy=np.nan)
        w = pref["weights"]
        churn_hist = [p[0][3] for p in tr_pairs]
        churn_tol = float(np.quantile(churn_hist, 0.9)) if churn_hist else None
        acc_l = pairwise_accuracy(te_pairs, w) if te_pairs else np.nan
        acc_e = pairwise_accuracy(te_pairs, np.full(4, 0.25)) if te_pairs else np.nan
        pref_rows.append(dict(origin=k, n_train=len(tr_pairs), n_test=len(te_pairs), w_cost=w[0], w_schedule=w[1],
                              w_workload=w[2], w_churn=w[3], train_acc=pref["pairwise_accuracy"],
                              test_acc_learned=acc_l, test_acc_equal=acc_e))
        log(f"origin {k}: learnt weights {np.round(w, 3)}; held-out pairwise accuracy {acc_l:.3f} (equal weights {acc_e:.3f})")
        # ---------------- E6
        insts = [(j, tag, inst) for j, (e, _) in enumerate(test_eps) for tag, inst in instances(e)]
        res = Parallel(n_jobs=a.jobs)(delayed(run_algos)(inst, w, seeds, gens, churn_tol) for _, _, inst in insts)
        for (j, tag, inst), rows in zip(insts, res):
            for r in rows:
                algo_rows.append(dict(origin=k, episode=j, instance=tag, n_tasks=inst.n, **r))
        rng = np.random.default_rng(k)
        pick = insts[:: max(len(insts) // (3 if a.smoke else 10), 1)]
        dres = Parallel(n_jobs=a.jobs)(delayed(dirichlet_verdict)(inst, w, 5 if a.smoke else 30, 2 if a.smoke else 3,
                                                                   gens, np.random.default_rng(rng.integers(1e9)))
                                       for _, _, inst in pick)
        for (j, tag, inst), rows in zip(pick, dres):
            for r in rows:
                dir_rows.append(dict(origin=k, episode=j, instance=tag, **r))
        log(f"  origin {k}: {len(insts)} instances optimised")
    Pf, A, D = pd.DataFrame(pref_rows), pd.DataFrame(algo_rows), pd.DataFrame(dir_rows)
    save(Pf, "e7_preferences.csv")
    save(A, "e6_optimisers.csv")
    save(D, "e6_dirichlet.csv")
    print("\nE7 learnt preference weights:\n", Pf.round(3).to_string(index=False))
    A["unit"] = A.origin.astype(str) + "-" + A.episode.astype(str) + "-" + A.instance
    agg = A.groupby(["unit", "algo"])[["score", "HV", "IGDplus", "time_s", "conv_iter"]].mean().reset_index()
    print("\nE6 optimisers (mean over instances):\n", agg.groupby("algo")[["score", "HV", "IGDplus", "time_s", "conv_iter"]].mean().round(4).to_string())
    for metric, low in (("score", True), ("HV", False), ("IGDplus", True)):
        t = agg.pivot_table(index="unit", columns="algo", values=metric).dropna(axis=1, how="all")
        out = stats.compare(t, lower_is_better=low)
        print(f"\n[{metric}] Friedman p={out.get('friedman_p', float('nan')):.3g}; mean ranks:",
              {kk: round(v, 2) for kk, v in sorted(out["mean_rank"].items(), key=lambda x: x[1])})
        print(out["pairwise"][["a", "b", "median_diff", "p_holm", "A12", "magnitude"]].round(4).to_string(index=False))
        save(out["pairwise"], f"e6_pairwise_{metric}.csv")
    if len(D):
        print(f"\nDirichlet robustness: GA better than PSO in {D.ga_better.mean():.1%} of {len(D)} weight samples")


if __name__ == "__main__":
    main()
