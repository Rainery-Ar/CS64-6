# fedavg.py
from collections import OrderedDict

def fedavg(dicts):
    """
    state_dicts_with_weights: [(state_dict, weight_n), ...]
    """
    # initial
    agg = OrderedDict()
    total = sum(w for _, w in dicts)

    for i in dicts[0][0].keys():
        agg[i] = sum((sd[i] * (w / total) for sd, w in dicts))
    return agg
