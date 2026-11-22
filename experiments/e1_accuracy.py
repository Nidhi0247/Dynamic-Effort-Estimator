"""E1 (RQ1) estimation accuracy and E2 (RQ2) ablations on held-out, later data.

Every model sees the same per-project time splits (3 rolling origins), the same target and features,
and the same tuning budget on the validation window.  Unit of analysis for statistics = project
(metrics averaged over origins), compared with Friedman + Wilcoxon/Holm + A12.
"""
from common import args, dataset, fitted_model, log, origin_splits, save
import numpy as np
import pandas as pd

from dynest.baselines import all_baselines
from dynest.conformal import ConformalIntervals
from dynest.features import MEASURED
from dynest.evaluation import metrics
from dynest import stats

ABLATIONS = {
    "A1 no SOM": dict(use_som=False),
    "A2 no ANFIS": dict(use_anfis=False),
    "A3 no Mamdani": dict(mamdani="none"),
    "A4 expert Mamdani (PBL rules)": dict(mamdani="expert"),
    "A5 no learned attributes": dict(use_aux=False),
}
COCOMO = ["integration_hat", "clarity_hat", "familiarity_log", "priority_ord", "desc_words_log"]
CAEA_MAP = dict(configuration="integration_hat", ease_of_operation="clarity_hat", complex_processing="desc_words_log",
                data_transactions="n_links_t", performance="priority_ord")


def with_features(m, df):
    B = m.transform(df)
    d = df.reset_index(drop=True).copy()
    for c in B.columns:
        if c not in d.columns:
            d[c] = B[c].values
    d["hours"] = d.effort_h
    return d


def baseline_intervals(m, b, VA, TE, alpha=0.05, seed=0):
    """Same recipe as DynEst-R's S6 (normalised, Mondrian by SOM cluster, calibrated on the validation window),
    with the baseline's own log prediction as the difficulty feature in place of DynEst-R's c_hat."""
    cols = MEASURED + ["clarity_hat", "integration_hat"]
    pv, pt = b.predict(VA), b.predict(TE)
    Fv = np.column_stack([VA[cols].values, np.log(pv)])
    Ft = np.column_stack([TE[cols].values, np.log(pt)])
    order = np.argsort(VA.resolved_at.values, kind="stable")             # time-ordered calibration window
    ci = ConformalIntervals(alpha=alpha, seed=seed).fit(Fv[order], np.log(VA.hours.values / pv)[order],
                                                        VA.som_bmu.values[order])
    lo, hi, _ = ci.interval(pt, Ft, TE.som_bmu.values)
    return pt, lo, hi


def per_project(name, k, te, pred, tr, low=None, high=None):
    rows = []
    for p, idx in te.groupby("project_id").indices.items():
        r = metrics(te.effort_h.values[idx], pred[idx], tr.effort_h.values,
                    None if low is None else low[idx], None if high is None else high[idx])
        rows.append(dict(model=name, origin=k, project_id=p, **r))
    r = metrics(te.effort_h.values, pred, tr.effort_h.values, low, high)
    rows.append(dict(model=name, origin=k, project_id="ALL", **r))
    return rows


def main():
    a = args([(["--no-ablations"], dict(action="store_true"))])
    ds = dataset()
    log(f"projects={len(ds.projects)} rows={len(ds.rows)}")
    rows = []
    for k, (f, tr, va, te) in enumerate(origin_splits(a.smoke)):
        log(f"origin {k} ({f}): train={len(tr)} val={len(va)} test={len(te)}")
        m = fitted_model(k, tr, va, a.backend, a.smoke)
        p = m.predict(te)
        rows += per_project("DynEst-R (proposed)", k, te, p.hours.values, tr, p.low.values, p.high.values)
        feats = list(m.fb.columns) + ["clarity_hat", "integration_hat"] + m.prof_cols
        TR, VA, TE = with_features(m, tr), with_features(m, va), with_features(m, te)
        for b in all_baselines(feats, COCOMO, CAEA_MAP, n_trials=4 if a.smoke else 40, seed=k,
                               caea_constants=str(ds.store.root.parents[1] / "configs" / "caea_constants.yaml")):
            try:
                b.fit(TR, VA)
            except ValueError as e:
                log(f"  {b.name}: skipped ({e})")
                continue
            pt, lo, hi = baseline_intervals(m, b, VA, TE, seed=k)
            rows += per_project(b.name, k, te, pt, tr, lo, hi)
            log(f"  {b.name}: done")
        if not a.no_ablations:
            for name, cfg in ABLATIONS.items():
                am = fitted_model(k, tr, va, a.backend, a.smoke, cfg={**m.cfg, **cfg, "text_backend": a.backend},
                                  tag=name.split()[0])
                ap = am.predict(te)
                rows += per_project(name, k, te, ap.hours.values, tr, ap.low.values, ap.high.values)
                log(f"  {name}: done")
    res = pd.DataFrame(rows)
    from common import RESULTS
    for f in sorted(RESULTS.glob("e10_text_*_raw.csv")):                 # B8 fine-tuned text models (GPU run)
        if "smoke" not in f.name and not a.smoke:
            res = pd.concat([res, pd.read_csv(f)], ignore_index=True)
            log(f"merged {f.name}")
    res["project_id"] = res.project_id.astype(str)
    save(res, "e1_accuracy_raw.csv")
    allp = res[res.project_id == "ALL"].groupby("model")[["MAE", "MdAE", "LMAE", "SA", "MdMRE", "PRED25", "MIBRE",
                                                          "coverage", "rel_width", "interval_score_rel"]].mean()
    print("\nPooled test metrics (mean over origins):\n", allp.sort_values("MAE").round(3).to_string())
    proj = res[res.project_id != "ALL"].groupby(["project_id", "model"]).MAE.mean().unstack()
    for metric, low_better in (("MAE", True), ("LMAE", True), ("SA", False)):
        t = res[res.project_id != "ALL"].groupby(["project_id", "model"])[metric].mean().unstack()
        out = stats.compare(t, lower_is_better=low_better)
        pw = out["pairwise"]
        pw = pw[(pw.a == "DynEst-R (proposed)") | (pw.b == "DynEst-R (proposed)")]
        print(f"\n[{metric}] units={out['n_units']} Friedman p={out.get('friedman_p', float('nan')):.3g} "
              f"CD={out['cd_nemenyi']:.2f}")
        print("mean ranks:", {k_: round(v, 2) for k_, v in sorted(out["mean_rank"].items(), key=lambda x: x[1])})
        print(pw[["a", "b", "median_diff", "ci_low", "ci_high", "p_holm", "r_rb", "A12"]].round(3).to_string(index=False))
        save(out["pairwise"], f"e1_pairwise_{metric}.csv")
    save(allp.reset_index(), "e1_accuracy_summary.csv")
    return proj


if __name__ == "__main__":
    main()
