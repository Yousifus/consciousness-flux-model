# src/rich/hierarchy.py
import numpy as np

DEFAULT_LEVELS = {
    "quantum": 0.001,
    "cellular": 0.01,
    "neural": 0.1,
    "social": 1.0,
    "digital": 2.0,
    "cosmic": 10.0
}

def dynamic_level_weights(year, base=None):
    base = dict(DEFAULT_LEVELS if base is None else base)
    # gentle evolution (plants/digital recognition rising)
    evol = (year - 1950) / 75.0
    base["digital"] *= (1.0 + 4.0*np.clip(evol,0,1))
    base["cellular"]*= (1.0 + 0.4*np.clip(evol,0,1))
    return base
