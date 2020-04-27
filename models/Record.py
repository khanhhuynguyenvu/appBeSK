import numpy as np


def default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


class Record:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values

    def toDictionary(self):
        d = {}
        for key, value in zip(self.keys, self.values):
            d[key] = default(value)
        return d
