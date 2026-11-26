"""Builds the practitioner-study kit (paper/user_study/) from the trained transparent model DynEst-R-T.

Each stimulus is a real held-out requirement (origin 0 test window) with the model's estimate, its 95% interval and
the rules that fired, in plain language.  Half of the stimuli are shown with the explanation and half without
(counter-balanced across two participant groups), so the study measures whether the explanation changes how well
people judge an estimate, not just whether they like it.
"""
import pickle

from common import MODELS, ROOT, origin_splits
import numpy as np
import pandas as pd

NAMES = {"som_prior": "hours per story point of similar past work in this project",
         "familiarity_log": "the assignee's familiarity with these components",
         "log_sp": "the story-point estimate"}
TERM = {"very_low": "very low", "low": "low", "fairly_low": "fairly low", "medium": "medium",
        "fairly_high": "fairly high", "high": "high", "very_high": "very high"}


def plain(rule):
    """IF som_prior IS high AND ... THEN effort IS high  ->  readable sentence."""
    cond, cons = rule[3:].split(" THEN effort IS ")
    parts = []
    for c in cond.split(" AND "):
        var, term = c.split(" IS ")
        parts.append(f"{NAMES.get(var, var)} is {TERM.get(term, term)}")
    return "If " + ", and ".join(parts) + f", then effort per story point is {TERM.get(cons, cons)}."


def main(n=12, seed=3):
    f, tr, va, te = origin_splits()[0]
    m = pickle.loads((MODELS / "dynestr_o0_minilm_b40_T.pkl").read_bytes())
    rng = np.random.default_rng(seed)
    pool = te[(te.sp >= 1) & (te.text.str.len() > 120)].reset_index(drop=True)
    pick = pool.iloc[rng.choice(len(pool), n, replace=False)].reset_index(drop=True)
    pred = m.predict(pick)
    out = ROOT / "paper" / "user_study"
    out.mkdir(parents=True, exist_ok=True)
    lines, key = ["# Stimuli (DynEst-R-T, real held-out TAWOS requirements)\n"], []
    for i, r in pick.iterrows():
        p = pred.iloc[i]
        ex = m.explain(pick.iloc[[i]], top=3)
        group_a_sees = "explanation" if i % 2 == 0 else "no explanation"
        lines += [f"## Item {i + 1}  (group A: {group_a_sees}; group B: the other condition)\n",
                  f"**Type / priority / story points:** {r.issuetype_t} / {r.priority_t} / {r.sp:g}\n",
                  f"**Requirement (as written when work started):** {str(r.text)[:700]}\n",
                  f"**Model estimate:** {p.hours:.1f} hours (95% range {p.low:.1f} to {p.high:.1f} hours)\n",
                  "**Why (rules that fired, strongest first):**\n"]
        lines += [f"- {plain(x['rule'])} (strength {x['firing_strength']:.2f}, seen in {x['support']} past items)"
                  for x in ex.get("rules", [])]
        lines.append("")
        key.append(dict(item=i + 1, issue_key=r.get("key", r.issue_id), actual_hours=round(float(r.effort_h), 1),
                        estimate=round(float(p.hours), 1), within_interval=bool(p.low <= r.effort_h <= p.high)))
    (out / "stimuli.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(key).to_csv(out / "answer_key_DO_NOT_SHOW.csv", index=False)
    print("wrote", out / "stimuli.md", "and the answer key")


if __name__ == "__main__":
    main()
