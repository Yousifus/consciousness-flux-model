# src/rich/synthetic_rich_data.py
import numpy as np
import pandas as pd

def logistic(x, x0, k=0.2, L=1.0):
    return L / (1.0 + np.exp(-k*(x - x0)))

def bass_t(t, p=0.03, q=0.38, m=1.0):
    # Bass diffusion cumulative adoption (scaled to m)
    t = np.asarray(t, float)
    adopters = (m*(1-np.exp(-(p+q)*t)))/(1+(q/p)*np.exp(-(p+q)*t))
    return np.clip(adopters/m, 0, 1)

def gaussian_copula(n, corr, seed=1337):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, corr.shape[0]))
    g = z @ L.T
    u = 0.5*(1+erf(g/np.sqrt(2)))
    return np.clip(u, 1e-6, 1-1e-6)

def erf(x):
    # minimal error function via numerical approximation
    # Abramowitz & Stegun 7.1.26
    a1,a2,a3,a4,a5 = 0.254829592,-0.284496736,1.421413741,-1.453152027,1.061405429
    t = 1.0/(1.0+0.3275911*np.abs(x))
    y = 1.0 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*np.exp(-x*x)
    return np.sign(x)*y

def rich_synthetic(years, seed=1337):
    """Return a DataFrame with richer correlated drivers & shocks."""
    rng = np.random.default_rng(seed)
    years = np.asarray(years)
    n = len(years)
    t = years - years[0]

    # Base scalars (rough magnitudes)
    internet = logistic(years, x0=2003, k=0.25)         # internet_penetration_pct (0..1)
    social   = logistic(years, x0=2012, k=0.30)
    oss      = logistic(years, x0=2008, k=0.22)
    vr_ar    = logistic(years, x0=2020, k=0.20)
    bci      = logistic(years, x0=2030, k=0.15)         # stays small in current window
    creators = 0.1 + 0.9*oss                             # opensource_contributors proxy
    ai_rel   = logistic(years, x0=2022, k=0.6)          # ai_models_released (scaled)

    # Connectivity split (developed vs developing)
    L_dev = 0.8 + 4.2/(1+np.exp(-0.08*(years-1990)))
    L_dev += rng.normal(0,0.05,size=n)
    L_dev = np.clip(L_dev, 0.5, 5.0)
    L_devp = 0.5 + 3.8/(1+np.exp(-0.06*(years-2000)))
    L_devp += rng.normal(0,0.07,size=n)
    L_devp = np.clip(L_devp, 0.3, 4.5)

    # Culture/quality proxies
    education_years = 4 + 8*logistic(years, x0=1995, k=0.06)
    philosophy_pubs = 1 + 9*logistic(years, x0=2000, k=0.08)
    meditation_m = 5 + 50*logistic(years, x0=2010, k=0.10)
    psychedelic_rate = 0.01 + 0.04*logistic(years, x0=2018, k=0.35)

    # Mystical shock table (multipliers on quality)
    shock = np.ones(n)
    shock[years==1969] *= 1.15
    shock[years==2001] *= 0.92
    shock[years==2020] *= 1.25
    shock[years==2022] *= 1.40

    # Correlation between quality proxies (education, pubs, meditation)
    corr = np.array([
        [1.0, 0.7, 0.5],
        [0.7, 1.0, 0.45],
        [0.5, 0.45, 1.0]
    ])
    u = gaussian_copula(n, corr, seed=seed+7)
    # Map u to scaled ranges around the base trends
    edu = education_years * (0.9 + 0.2*u[:,0])
    phil= philosophy_pubs * (0.9 + 0.2*u[:,1])
    med = meditation_m * (0.9 + 0.2*u[:,2])

    df = pd.DataFrame({
        "year": years,
        # technology adoption curves
        "internet_penetration": internet,
        "social_media_users": social,
        "opensource_contributors": creators,
        "ai_models_released": ai_rel,
        "vr_ar_users": vr_ar,
        "bci_index": bci,
        # connectivity split
        "L_index_developed": L_dev,
        "L_index_developing": L_devp,
        # culture/quality
        "avg_education_years": edu,
        "philosophy_publications_index": phil,
        "meditation_practitioners_millions": med,
        "psychedelic_usage_rate": psychedelic_rate,
        "mystical_shock_multiplier": shock
    })
    return df
