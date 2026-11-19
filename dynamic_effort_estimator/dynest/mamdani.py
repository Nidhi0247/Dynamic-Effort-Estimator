"""Custom Mamdani Fuzzy Inference System (min-AND, min-implication, max-aggregation, centroid)."""
import numpy as np


def trap(x, a, b, c, d):
    x = np.asarray(x, float)
    left = (x - a) / (b - a) if b > a else (x >= a).astype(float)
    right = (d - x) / (d - c) if d > c else (x <= d).astype(float)
    return np.clip(np.minimum(left, right), 0, 1)


def _scale5(names): 
    pts = [(0, 0, 1, 2.5), (1, 2.5, 2.5, 5), (2.5, 5, 5, 7.5), (5, 7.5, 7.5, 9), (7.5, 9, 10, 10)]
    return dict(zip(names, pts))


INPUTS = {
    "complexity": _scale5(["very_low", "low", "medium", "high", "very_high"]),
    "clarity": _scale5(["very_vague", "vague", "moderate", "clear", "very_clear"]),
    "experience": _scale5(["very_low", "low", "medium", "high", "very_high"]),
}
OUTPUT_UNIVERSE = (0.0, 3.0)   
_OUT_NAMES = ["extremely_low", "very_low", "low", "fairly_low", "medium", "fairly_high", "high", "very_high",
              "extremely_high"]
_STEP = 3.0 / (len(_OUT_NAMES) - 1)
OUTPUT_TERMS = {n: (max(k * _STEP - _STEP, 0), k * _STEP, k * _STEP, min(k * _STEP + _STEP, 3.0))
                for k, n in enumerate(_OUT_NAMES)}


_COMPLEXITY_BASE = [0.10, 0.475, 0.85, 1.225, 1.60]     
_CLARITY_MULT = [1.40, 1.30, 1.20, 1.10, 1.00]           
_EXPERIENCE_MULT = [1.30, 1.18, 1.08, 1.00, 0.92]       


def build_rules():
    cl, ca, ex = list(INPUTS["complexity"]), list(INPUTS["clarity"]), list(INPUTS["experience"])
    centres = np.array([t[1] for t in OUTPUT_TERMS.values()])
    rules = []
    for i, c in enumerate(cl):
        for j, k in enumerate(ca):
            for m_, e in enumerate(ex):
                v = _COMPLEXITY_BASE[i] * _CLARITY_MULT[j] * _EXPERIENCE_MULT[m_]
                rules.append(dict(complexity=c, clarity=k, experience=e,
                                  effort=_OUT_NAMES[int(np.argmin(np.abs(centres - v)))]))
    return rules


def rule_text(r):
    return (f"IF complexity IS {r['complexity']} AND clarity IS {r['clarity']} "
            f"AND experience IS {r['experience']} THEN effort IS {r['effort']}")


class MamdaniFIS:
    def __init__(self, rules=None, resolution=301):
        self.rules = rules or build_rules()
        self.y = np.linspace(*OUTPUT_UNIVERSE, resolution)
        self.out_mu = {t: trap(self.y, *p) for t, p in OUTPUT_TERMS.items()}
        self.terms = list(OUTPUT_TERMS)

    def memberships(self, var, x):
        return {t: trap(x, *p) for t, p in INPUTS[var].items()}

    def infer_batch(self, complexity, clarity, experience):
        """Returns (crisp (N,), rule strengths (N,R), aggregated curve (N,Y))."""
        vals = dict(complexity=np.atleast_1d(np.asarray(complexity, float)),
                    clarity=np.atleast_1d(np.asarray(clarity, float)),
                    experience=np.atleast_1d(np.asarray(experience, float)))
        mem = {v: self.memberships(v, vals[v]) for v in INPUTS}
        n = len(vals["complexity"])
        strengths = np.zeros((n, len(self.rules)))
        for ri, r in enumerate(self.rules):
            strengths[:, ri] = np.minimum.reduce([mem[v][r[v]] for v in INPUTS])      
        term_s = np.zeros((n, len(self.terms)))
        for ri, r in enumerate(self.rules):
            ti = self.terms.index(r["effort"])
            term_s[:, ti] = np.maximum(term_s[:, ti], strengths[:, ri])
        agg = np.zeros((n, len(self.y)))
        for ti, t in enumerate(self.terms):                                          
            agg = np.maximum(agg, np.minimum(term_s[:, ti:ti + 1], self.out_mu[t][None, :]))
        area = agg.sum(1)
        crisp = np.where(area > 1e-12, (agg * self.y).sum(1) / np.maximum(area, 1e-12), self.y.mean())
        return crisp, strengths, agg

    def explain(self, complexity, clarity, experience, top=5, min_strength=1e-3):
        crisp, s, agg = self.infer_batch(complexity, clarity, experience)
        order = np.argsort(-s[0])
        fired = [dict(rule=rule_text(self.rules[i]), firing_strength=float(s[0, i]))
                 for i in order[:top] if s[0, i] > min_strength]
        return float(crisp[0]), fired, agg[0]
