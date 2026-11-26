"""E11: main TAWOS comparison with an EQUAL tuning budget for every model, plus the transparent variant and
interpretable baselines.  Addresses three review points:
  * E1 gave DynEst-R 8 random-search trials against up to 40 for baselines -> here every tuned model gets N (40);
  * the Mamdani stage of DynEst-R reads a latent score (c_hat) -> DynEst-R-T uses the SOM prior plus the two
    measured features most related to the target (chosen on the training window), so every rule is readable;
  * an interpretability claim needs interpretable competitors -> shallow regression tree (depth <= 4) and EBM.
Same splits, target, features, back-transform and conformal recipe as E1.  Rows use E1's raw format.

The expected direction of each readable input is declared HERE, before any result was inspected, and is only
used to score the learnt rules (never in training):
"""
import pickle

from common import MODELS, args, dataset, log, origin_splits, save
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from dynest import tune
from dynest.baselines import all_baselines, interpretable_baselines
from e1_accuracy import CAEA_MAP, COCOMO, baseline_intervals, per_project, with_features

EXPECTED = {"som_prior": +1, "familiarity_log": -1, "tenure_log": -1, "n_links_t": +1, "n_components_t": +1,
            "desc_words_log": +1, "ac_lines": +1, "comments_t": +1, "log_sp": -1}
FULL, TRANSPARENT = "DynEst-R (proposed)", "DynEst-R-T (proposed, transparent)"


def origin_job(k, tr, va, te, trials, backend):
    import warnings
    warnings.filterwarnings("ignore")
    comps = dataset().components
    rows, out = [], {}
    for name, cfg in ((FULL, {}), (TRANSPARENT, dict(mamdani_inputs="auto", use_anfis=False))):
        m = tune(tr, va, comps, n_trials=trials, base_cfg=dict(text_backend=backend, **cfg), seed=k,
                 log=lambda *a: None)
        (MODELS / f"dynestr_o{k}_{backend}_b{trials}_{'T' if 'transparent' in name else 'full'}.pkl").write_bytes(
            pickle.dumps(m))
        p = m.predict(te)
        rows += per_project(name, k, te, p.hours.values, tr, p.low.values, p.high.values)
        out[name] = m
        log(f"origin {k}: {name} tuned ({trials} trials), Mamdani inputs {m.mcols}, val MAE {m.val_mae:.2f}")
    m = out[FULL]
    feats = list(m.fb.columns) + ["clarity_hat", "integration_hat"] + m.prof_cols
    TR, VA, TE = with_features(m, tr), with_features(m, va), with_features(m, te)
    for b in all_baselines(feats, COCOMO, CAEA_MAP, n_trials=trials, seed=k) + \
            interpretable_baselines(feats, n_trials=trials, seed=k):
        try:
            b.fit(TR, VA)
        except ValueError as e:
            log(f"origin {k}: {b.name} skipped ({e})")
            continue
        pt, lo, hi = baseline_intervals(m, b, VA, TE, seed=k)
        rows += per_project(b.name, k, te, pt, tr, lo, hi)
        log(f"origin {k}: {b.name} done")
    # readable rule base of the transparent model + domain consistency against EXPECTED
    t = out[TRANSPARENT]
    rules = t.fis.rule_table().assign(origin=k, inputs=", ".join(t.mcols))
    rr = {r["ante"]: r["cons"] for r in t.fis.rules}
    mono = []
    for i, nm in enumerate(t.mcols):
        if nm not in EXPECTED:
            continue
        pairs = [(rr[a], rr[b]) for a in rr for b in [a[:i] + (a[i] + 1,) + a[i + 1:]] if b in rr]
        d = np.sign([y - x for x, y in pairs]) * EXPECTED[nm]
        mono.append(dict(origin=k, input=nm, expected=EXPECTED[nm], pairs=len(pairs),
                         consistent=float(np.mean(d >= 0)) if pairs else np.nan,
                         opposite=float(np.mean(d < 0)) if pairs else np.nan))
    return rows, rules, mono


def main():
    a = args([(["--trials"], dict(type=int, default=40))])
    splits = origin_splits(a.smoke)
    trials = 3 if a.smoke else a.trials
    log(f"E11: {len(splits)} origins, {trials} trials for every tuned model")
    res = Parallel(n_jobs=min(a.jobs, len(splits)), verbose=10)(
        delayed(origin_job)(k, tr, va, te, trials, a.backend) for k, (f, tr, va, te) in enumerate(splits))
    R = pd.DataFrame([r for rows, _, _ in res for r in rows])
    R["project_id"] = R.project_id.astype(str)
    tag = "_smoke" if a.smoke else ""
    save(R, f"e11_equal_budget_raw{tag}.csv")
    save(pd.concat([r for _, r, _ in res], ignore_index=True), f"e11_transparent_rules{tag}.csv")
    M = pd.DataFrame([m for _, _, ms in res for m in ms])
    save(M, f"e11_transparent_monotonicity{tag}.csv")
    allp = R[R.project_id == "ALL"].groupby("model")[["MAE", "MdAE", "LMAE", "SA", "coverage", "rel_width",
                                                      "interval_score_rel"]].mean()
    print("\nPooled test metrics, equal budget (mean over origins):\n", allp.sort_values("MAE").round(3).to_string())
    print("\nTransparent rules: domain consistency\n", M.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
