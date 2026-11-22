"""E9: interpretability evidence for the trained DynEst-R models (TAWOS, three rolling origins).

Reported per origin: rule count, mean antecedent length, rule-base coverage of the test window, share of test
requirements explained by <= 3 rules (>= 80% of the firing mass), SOM quantisation / topographic error and
cluster sizes.  Across origins: Jaccard similarity of the linguistic rule sets and, for antecedents shared by
two origins, the share with the same consequent (rule stability).  Also writes the full rule tables.
"""
import itertools

from common import fitted_model, log, origin_splits, save
import numpy as np
import pandas as pd


def main():
    splits = origin_splits()
    rows, rules, models = [], [], {}
    for k, (f, tr, va, te) in enumerate(splits):
        m = fitted_model(k, tr, va, "minilm")
        models[k] = m
        fis = m.fis
        B = m.transform(te)
        X = B[m.mcols].values
        _, s, _ = fis.infer(X, return_strengths=True)
        mass = np.sort(s, 1)[:, ::-1]
        tot = mass.sum(1, keepdims=True) + 1e-12
        top3 = (mass[:, :3].sum(1) / tot[:, 0]) >= 0.8
        Zt = m.som_scaler.transform(B[m.som_cols].values)
        sizes = np.bincount(B.som_bmu.values, minlength=len(m.som.W))
        rows.append(dict(origin=k, n_rules=len(fis.rules), inputs=", ".join(fis.input_names),
                         terms=str(fis.n_terms), n_out=fis.n_out, possible_cells=int(np.prod(fis.n_terms)),
                         test_coverage=fis.coverage(X), explained_by_3_rules=float(top3.mean()),
                         median_support=float(np.median([r["support"] for r in fis.rules])),
                         som_grid=f"{m.som.rows}x{m.som.cols}", som_qe=m.som.quantization_error(Zt),
                         som_te=m.som.topographic_error(Zt), som_empty_neurons=int((sizes == 0).sum()),
                         som_largest_cluster_share=float(sizes.max() / sizes.sum())))
        t = fis.rule_table()
        t.insert(0, "origin", k)
        rules.append(t)
        log(f"origin {k}: {rows[-1]}")
    R = pd.DataFrame(rows)
    T = pd.concat(rules, ignore_index=True)
    save(R, "e9_interpretability.csv")
    save(T, "e9_rules.csv")

    stab = []
    for a, b in itertools.combinations(models, 2):
        fa, fb = models[a].fis, models[b].fis
        if fa.n_terms != fb.n_terms or fa.n_out != fb.n_out:
            stab.append(dict(a=a, b=b, comparable=False))
            continue
        ra = {r["ante"]: r["cons"] for r in fa.rules}
        rb = {r["ante"]: r["cons"] for r in fb.rules}
        full_a, full_b = set(ra.items()), set(rb.items())
        shared = set(ra) & set(rb)
        stab.append(dict(a=a, b=b, comparable=True, jaccard_rules=len(full_a & full_b) / len(full_a | full_b),
                         jaccard_antecedents=len(shared) / len(set(ra) | set(rb)),
                         same_consequent=float(np.mean([ra[x] == rb[x] for x in shared])) if shared else np.nan,
                         within_one_term=float(np.mean([abs(ra[x] - rb[x]) <= 1 for x in shared])) if shared else np.nan))
    S = pd.DataFrame(stab)
    save(S, "e9_rule_stability.csv")

    # domain consistency: neighbouring rules (one antecedent one term higher, rest equal) should move the
    # consequent up for complexity and not up for clarity / familiarity (expected signs, not used in training)
    expect = {"c_hat": +1, "clarity_hat": -1, "familiarity_log": -1}
    mono = []
    for k, m in models.items():
        rr = {r["ante"]: r["cons"] for r in m.fis.rules}
        for i, name in enumerate(m.fis.input_names):
            pairs = [(rr[a], rr[b]) for a in rr for b in [a[:i] + (a[i] + 1,) + a[i + 1:]] if b in rr]
            d = np.sign([y - x for x, y in pairs])
            mono.append(dict(origin=k, input=name, expected=expect.get(name, 0), pairs=len(pairs),
                             consistent=float(np.mean(d * expect.get(name, 0) >= 0)),
                             strictly=float(np.mean(d * expect.get(name, 0) > 0)), opposite=float(np.mean(d * expect.get(name, 0) < 0))))
    M = pd.DataFrame(mono)
    save(M, "e9_rule_monotonicity.csv")
    print("\nDomain consistency of neighbouring rules (share not violating the expected direction):")
    print(M.round(3).to_string(index=False))
    print(R.round(3).to_string(index=False))
    print("\nRule stability across origins:\n", S.round(3).to_string(index=False))
    print("\nExample rules (origin 0, highest support):")
    print(T[T.origin == 0].sort_values("support", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
