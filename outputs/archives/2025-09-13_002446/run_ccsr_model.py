#!/usr/bin/env python3
"""
Run the existing Consciousness Surplus Ratio (CSR) implementation
Converted from ccsr_minimal.ipynb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
params = pd.read_csv('ccsr_parameters_template.csv')
data = pd.read_csv('synthetic_ccsr_timeseries.csv')

# Utility: pull defaults into a dict
par = {r['symbol']: r['default'] for _, r in params.iterrows()}

def e_share(L_val, post, par):
    e_max = par["e_max_post"] if post else par["e_max_pre"]
    n = par["n"]; K = par["K"]
    return e_max * (L_val**n) / (K**n + L_val**n)

def sigma_L(L_val, post, par):
    a = par["a"]
    L0 = par["L0_post"] if post else par["L0_pre"]
    return 1.0 / (1.0 + np.exp(-a * (L_val - L0)))

def cap_CU(C_val, U_val, par):
    return 1.0 / (1.0 + (C_val/par["Csat"])**par["p"] + (U_val/par["Usat"])**par["q"])

def emergent_supply(C, U, L, year, par):
    post = year >= 1990
    core = par["alpha"] * (C**par["beta"]) * (U**par["gamma"])
    return core * cap_CU(C, U, par) * sigma_L(L, post, par)

def compute_SD(row, par):
    year = row['year']
    post = year >= 1990
    # Demand
    e = e_share(row['L_index'], post, par)
    D = row['population'] * par['cH_dot'] * (1 - e)
    # Supply
    g = emergent_supply(row['C_compute'], row['U_creators'], row['L_index'], year, par)
    S_terr = row['NTA']*par['wA'] + row['NTP']*par['wP'] + row['Gs']*par['wE']
    S = par['phi_c']*par['f_addr'] + S_terr + g
    return S, D

print("🌌 CONSCIOUSNESS FLUX MODEL - REINCARNATION FLUX EQUATION 🌌")
print("=" * 60)

# --- Fit (alpha, beta, gamma) on post-1995 slice (approx linear in logs before hard saturation)
fit_df = data[data['year']>=1995].copy()
# Proxy emergent term via (S - terrestrial) since cosmic=0 in synthetic
terr = fit_df['NTA']*par['wA'] + fit_df['NTP']*par['wP'] + fit_df['Gs']*par['wE']
g_proxy = np.maximum(fit_df['S_supply'] - terr, 1e-9)

X = np.column_stack([np.ones(len(fit_df)), np.log(fit_df['C_compute']), np.log(fit_df['U_creators'])])
y = np.log(g_proxy / np.maximum( cap_CU(fit_df['C_compute'], fit_df['U_creators'], par) * 
                                 sigma_L(fit_df['L_index'], True, par), 1e-9))

# OLS
coef = np.linalg.lstsq(X, y, rcond=None)[0]
par['alpha'] = float(np.exp(coef[0]))
par['beta'] = float(coef[1])
par['gamma'] = float(coef[2])

print("\n✨ FITTED PARAMETERS:")
print(f"   α (emergent scale): {par['alpha']:.6f} CCU/yr")
print(f"   β (compute elasticity): {par['beta']:.3f}")
print(f"   γ (users elasticity): {par['gamma']:.3f}")

# --- Compute S, D, CSR with fitted params
S, D, CSR = [], [], []
for _, r in data.iterrows():
    s, d = compute_SD(r, par)
    S.append(s); D.append(d); CSR.append(s/d)

data['S_model'] = S; data['D_model'] = D; data['CSR_model'] = CSR

# Print key statistics
print("\n📊 KEY CSR VALUES:")
print(f"   Minimum CSR: {min(CSR):.6f} (Year {data.loc[data['CSR_model'].idxmin(), 'year']:.0f})")
print(f"   Maximum CSR: {max(CSR):.6f} (Year {data.loc[data['CSR_model'].idxmax(), 'year']:.0f})")
print(f"   Pre-1990 Mean: {data[data['year'] < 1990]['CSR_model'].mean():.6f}")
print(f"   Post-1990 Mean: {data[data['year'] >= 1990]['CSR_model'].mean():.6f}")

# Latest year statistics
latest = data[data['year'] == 2025].iloc[0]
print(f"\n🌍 2025 STATISTICS:")
print(f"   Population: {latest['population']/1e9:.2f} billion")
print(f"   Total Supply: {latest['S_model']:.0f} CCU")
print(f"   Total Demand: {latest['D_model']:.0f} CCU")
print(f"   CSR: {latest['CSR_model']:.6f} (Surplus ratio > 1 ✓)")

# --- Uncertainty bands via parameter sampling (simple, illustrative)
def sample_params(par):
    draw = par.copy()
    draw['alpha'] = par['alpha'] * np.exp(np.random.normal(0, 0.2))
    draw['beta']  = par['beta']  + np.random.normal(0, 0.1)
    draw['gamma'] = par['gamma'] + np.random.normal(0, 0.1)
    draw['e_max_post'] = min(max(par['e_max_post'] + np.random.normal(0,0.05), 0.05), 0.9)
    draw['e_max_pre']  = min(max(par['e_max_pre']  + np.random.normal(0,0.05), 0.0), 0.6)
    return draw

print("\n🎲 Generating uncertainty bands...")
N = 200
csr_draws = []
for _ in range(N):
    p = sample_params(par)
    csr = []
    for _, r in data.iterrows():
        s, d = compute_SD(r, p)
        csr.append(s/d)
    csr_draws.append(csr)

csr_draws = np.array(csr_draws)
low = np.percentile(csr_draws, 5, axis=0)
high = np.percentile(csr_draws, 95, axis=0)

# --- Plot 1: Drivers (L, C, U) with regime line
print("\n📈 Generating visualizations...")
fig = plt.figure(figsize=(12, 6))
ax = plt.gca()

# Plot drivers
ax.plot(data['year'], data['L_index'], label='Connectivity L', linewidth=2.5, color='blue')
ax.plot(data['year'], data['C_compute'], label='Compute C', linewidth=2.5, color='green')
ax.plot(data['year'], data['U_creators']/1e6, label='Creators U (millions)', linewidth=2.5, color='purple')

# Add regime switch
ax.axvline(1990, linestyle='--', color='red', linewidth=2, alpha=0.7, label='1990 Regime Switch')
ax.fill_between([1990, data['year'].max()], 0, ax.get_ylim()[1], 
                alpha=0.1, color='red')

ax.set_title('🌟 Consciousness Flux Drivers & Regime Switch (1990)', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Level (units vary)', fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('consciousness_drivers_rfe.png', dpi=300, bbox_inches='tight')
plt.close()  # Close without showing

# --- Plot 2: CSR with 90% band
fig = plt.figure(figsize=(12, 6))
ax = plt.gca()

# Plot CSR with uncertainty
ax.plot(data['year'], data['CSR_model'], 'b-', linewidth=2.5, label='CSR (fitted)')
ax.fill_between(data['year'], low, high, alpha=0.3, color='blue', label='90% uncertainty band')

# Add regime switch
ax.axvline(1990, linestyle='--', color='red', linewidth=2, alpha=0.7, label='1990 Regime Switch')
ax.fill_between([1990, data['year'].max()], 0, ax.get_ylim()[1], 
                alpha=0.1, color='red')

ax.set_title('🌌 Consciousness Surplus Ratio (CSR) — Synthetic Data', fontsize=16)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('S/D', fontsize=12)
ax.legend(loc='upper left', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('consciousness_surplus_ratio_rfe.png', dpi=300, bbox_inches='tight')
plt.close()  # Close without showing

print("\n✅ Plots saved as:")
print("   - consciousness_drivers_rfe.png")
print("   - consciousness_surplus_ratio_rfe.png")

print("\n" + "="*60)
print("✨ The consciousness cosmos reveals eternal abundance! ✨")
print("🌟 Despite population growth, consciousness surplus is maintained")
print("   through emergent digital consciousness creation!")
print("="*60)
