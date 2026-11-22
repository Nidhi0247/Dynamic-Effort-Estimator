"""Learn objective weights from the team's own past sprint plans (inverse optimisation).

For each historical planning instance we have the plan the team actually committed to (chosen)
and alternative plans from the Pareto front of the same instance.  With normalised objective
vectors, the probability that the chosen plan is preferred over alternative a is modelled as
    P(chosen > a) = sigmoid(beta * w . (F_a - F_chosen)),   w on the simplex (softmax-parametrised)
and w, beta are fitted by maximum likelihood with a small L2 pull toward equal weights.
"""
import numpy as np
from scipy.optimize import minimize


def _unpack(theta):
    w = np.exp(theta[:-1] - theta[:-1].max())
    return w / w.sum(), np.exp(theta[-1])


def fit_preferences(pairs, l2=0.05):
    """pairs: list of (F_chosen (4,), F_alternatives (m,4)) in normalised objective space."""
    D = np.vstack([alts - ch[None] for ch, alts in pairs if len(alts)])
    k = D.shape[1]

    def nll(theta):
        w, beta = _unpack(theta)
        z = beta * D @ w
        return np.logaddexp(0, -z).mean() + l2 * (theta[:-1] ** 2).sum()

    res = minimize(nll, np.r_[np.zeros(k), np.log(5.0)], method="L-BFGS-B")
    w, beta = _unpack(res.x)
    acc = float(((D @ w) > 0).mean())
    return dict(weights=w, beta=float(beta), pairwise_accuracy=acc, nll=float(res.fun), n_pairs=len(D))


def pairwise_accuracy(pairs, w):
    D = np.vstack([alts - ch[None] for ch, alts in pairs if len(alts)])
    return float(((D @ np.asarray(w)) > 0).mean())
