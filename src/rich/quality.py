# src/rich/quality.py
import numpy as np

def bounded_index(z, lo=0.8, hi=1.3):
    z = np.asarray(z, float)
    z_sc = (z - z.mean())/(z.std()+1e-9)
    q = 1.0 + 0.15*z_sc  # scale -> ~ +/- 0.15
    return np.clip(q, lo, hi)

def quality_multiplier(df):
    # Combine education, philosophy pubs, meditation, psychedelic rate, shocks
    # Weighted z-sum → multiplier in [0.8, 1.3], then apply shock
    import numpy as np
    comp = (
        0.35*df["avg_education_years"].values +
        0.30*df["philosophy_publications_index"].values +
        0.25*df["meditation_practitioners_millions"].values +
        0.10*df["psychedelic_usage_rate"].values
    )
    q = bounded_index(comp, lo=0.85, hi=1.25)
    return q * df["mystical_shock_multiplier"].values
