"""Back-transform from log space, learnt on validation.

For a log-scale model z, predictions are base * exp(z + c).  The shift c that minimises validation
MAE in hours is the weighted median of log(actual / (base*exp(z))) with weights base*exp(z).
(Duan smearing would make predictions mean-unbiased, which with heavy-tailed effort inflates every
prediction several-fold and is the wrong target for absolute-error criteria.)
"""
import numpy as np


def mae_shift(actual, pred_unshifted):
    a, p = np.asarray(actual, float), np.asarray(pred_unshifted, float)
    r = np.log(a / p)
    o = np.argsort(r)
    w = np.cumsum(p[o])
    return float(r[o][np.searchsorted(w, 0.5 * w[-1])])
