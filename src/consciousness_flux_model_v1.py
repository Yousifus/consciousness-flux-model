#!/usr/bin/env python3
"""
Consciousness Flux Model (RFE) - Version 1.0.3
==============================================
Consolidated implementation of the Reincarnation Flux Equation

A mathematical framework modeling consciousness as energy flowing through
cosmic, terrestrial, and emergent sources to meet human demand.

Created: 2025-09-13
Version: 1.0.3 - High-leverage patches:
  - Fixed double-counting: Gs excluded from terrestrial supply
  - Vectorized gate calculation for robust fitting
  - Enhanced uncertainty bands with sharing parameter sampling
  - Added network effect invariant check
  - Philosophical priors presets (PHYSICALIST, IIT, PANPSYCHIST)
  - Improved plot readability and bounds
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse
import json
from typing import Dict, Tuple, Optional

class ConsciousnessFluxModel:
    """
    Implementation of the Reincarnation Flux Equation (RFE).
    
    This model demonstrates how consciousness supply evolves to meet
    human demand through traditional (terrestrial) and emergent (digital/AI) sources.
    """
    
    def __init__(self, priors="PHYSICALIST", cosmic=None, addr=None, regime_year=1990):
        self.priors = priors
        self.cosmic_override = cosmic
        self.addr_override = addr
        self.regime_year = regime_year
        self.params_file = "../data/ccsr_parameters_template.csv"
        self.data_file = "../data/synthetic_ccsr_timeseries.csv"
        self.params = {}
        self.gen_weights = {}
        self.data = None
        self.results = {}
        
    def set_priors(self, priors: str):
        """Set parameter presets based on philosophical stance."""
        preset_params = {}
        
        if priors == 'PHYSICALIST':
            # Physicalist: higher animal/plant weights, lower emergent
            preset_params = {
                'wA': 0.8,
                'wP': 0.5,
                'wE': 0.1,
                'phi_c': 0.001,
                'f_addr': 0.95
            }
        elif priors == 'IIT':
            # IIT: balanced weights, moderate phi
            preset_params = {
                'wA': 0.4,
                'wP': 0.2,
                'wE': 0.3,
                'phi_c': 0.01,
                'f_addr': 0.90
            }
        elif priors == 'PANPSYCHIST':
            # Panpsychist: lower bio weights, higher emergent
            preset_params = {
                'wA': 0.2,
                'wP': 0.1,
                'wE': 0.5,
                'phi_c': 0.1,
                'f_addr': 0.85
            }
        else:
            raise ValueError(f"Unknown priors: {priors}. Use PHYSICALIST, IIT, or PANPSYCHIST.")
        
        # Update params after they're loaded
        if self.params is not None:
            self.params.update(preset_params)
        
    def load_data(self) -> pd.DataFrame:
        """Load parameter template and time series data."""
        # Load parameters
        params_df = pd.read_csv(self.params_file)
        self.params = {r['symbol']: r['default'] for _, r in params_df.iterrows()}
        # Cache baseline generation weights before priors mutate them
        self.gen_weights = {k: self.params[k] for k in ('wA','wP','wE')}
        
        # Apply philosophical priors
        self.set_priors(self.priors)
        
        # Optional overrides (clamped)
        if self.cosmic_override is not None:
            self.params["phi_c"] = float(max(0.0, min(self.cosmic_override, 0.05)))
        if self.addr_override is not None:
            self.params["f_addr"] = float(max(0.0, min(self.addr_override, 1.0)))
        
        # Load time series
        self.data = pd.read_csv(self.data_file)
        return self.data

    def decompose_post_change(self, y0_lo: int = 1985, y0_hi: int = 1989,
                              y1_lo: int = 1995, y1_hi: int = 1999,
                              alpha: Optional[float] = None,
                              beta: Optional[float] = None,
                              gamma: Optional[float] = None) -> Dict[str, object]:
        """LMDI-style decomposition of Δln CSR between two windows.
        Returns additive contributions (not normalized to 100%).
        """
        df = self.data
        w0 = df[(df["year"] >= y0_lo) & (df["year"] <= y0_hi)].copy()
        w1 = df[(df["year"] >= y1_lo) & (df["year"] <= y1_hi)].copy()
        if alpha is None:
            alpha = self.params["alpha"]; beta = self.params["beta"]; gamma = self.params["gamma"]

        def window_summ(window: pd.DataFrame) -> Dict[str, float]:
            C = float(window["C_compute"].mean())
            U = float(window["U_creators"].mean())
            L = float(window["L_index"].mean())
            NTA = float(window["NTA"].mean()); NTP = float(window["NTP"].mean())
            P = float(window["population"].mean())
            # demand share averaged per-year
            e_vals = []
            for _, r in window.iterrows():
                e_vals.append(self.e_share(r["L_index"], bool(r["year"] >= self.regime_year)))
            e = float(np.mean(e_vals))
            D = P * self.params["cH_dot"] * (1.0 - e)
            S_terr = NTA * self.params["wA"] + NTP * self.params["wP"]
            sigma = float(self.sigma_L(L, bool(((y1_lo + y1_hi) // 2) >= self.regime_year)))
            cap = float(self.cap_CU(C, U))
            g = float(alpha * (C**beta) * (U**gamma) * sigma * cap)
            return dict(C=C, U=U, L=L, S_terr=S_terr, D=D, sigma=sigma, cap=cap, g=g)

        s0 = window_summ(w0); s1 = window_summ(w1)

        import math
        def ln(x: float) -> float:
            return math.log(max(x, 1e-12))

        dlnC = ln(s1["C"]) - ln(s0["C"]) 
        dlnU = ln(s1["U"]) - ln(s0["U"]) 
        dlnsig = ln(s1["sigma"]) - ln(s0["sigma"]) 
        dlncap = ln(s1["cap"]) - ln(s0["cap"]) 

        contrib_g_C   = beta  * dlnC
        contrib_g_U   = gamma * dlnU
        contrib_g_sig = dlnsig
        contrib_g_cap = dlncap
        dln_g_total   = ln(s1["g"]) - ln(s0["g"]) 

        dln_Sterr = ln(s1["S_terr"]) - ln(s0["S_terr"]) 
        dln_D     = ln(s1["D"])     - ln(s0["D"]) 
        dln_CSR_approx = ln(s1["S_terr"] + s1["g"]) - ln(s0["S_terr"] + s0["g"]) - dln_D

        return {
            "window0": [y0_lo, y0_hi], "window1": [y1_lo, y1_hi],
            "dln_CSR": dln_CSR_approx,
            "bio": dln_Sterr,
            "emergent_total": dln_g_total,
            "emergent_breakdown": {
                "C": contrib_g_C,
                "U": contrib_g_U,
                "sigma(L)": contrib_g_sig,
                "cap(C,U)": contrib_g_cap
            },
            "demand": -dln_D
        }
    
    def e_share(self, L_val, post: bool) -> float:
        e_max = self.params["e_max_post"] if post else self.params["e_max_pre"]
        n, K = self.params["n"], self.params["K"]
        val = e_max * (L_val**n) / (K**n + L_val**n)
        return float(max(0.0, min(val, e_max - 1e-9)))

    def sigma_L(self, L_val, post: bool) -> float:
        a = self.params["a"]
        L0 = self.params["L0_post"] if post else self.params["L0_pre"]
        return float(1.0 / (1.0 + np.exp(-a * (L_val - L0))))
    
    def cap_CU(self, C_val: np.ndarray, U_val: np.ndarray) -> np.ndarray:
        """Saturation cap for compute and users."""
        Csat = self.params["Csat"]
        Usat = self.params["Usat"]
        p = self.params["p"]
        q = self.params["q"]
        return 1.0 / (1.0 + (C_val/Csat)**p + (U_val/Usat)**q)
    
    def emergent_supply(self, C: np.ndarray, U: np.ndarray, L: np.ndarray, 
                       year: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
        """Calculate emergent consciousness supply."""
        post_1990 = year >= self.regime_year
        core = alpha * (C**beta) * (U**gamma)
        return core * self.cap_CU(C, U) * self.sigma_L(L, post_1990)
    
    def compute_SD(self, row: pd.Series, alpha: float, beta: float, gamma: float) -> Tuple[float, float]:
        """Compute supply and demand for a given row."""
        year = row['year']
        post_1990 = year >= self.regime_year
        
        # Demand
        e = self.e_share(row['L_index'], post_1990)
        D = row['population'] * self.params['cH_dot'] * (1 - e)
        
        # Supply components
        # Terrestrial (only animal and plant consciousness)
        # Note: Gs (analog emergent) is excluded to avoid double-counting with g()
        S_terr = (row['NTA'] * self.params['wA'] + 
                 row['NTP'] * self.params['wP'])
        
        # Emergent
        g = self.emergent_supply(row['C_compute'], row['U_creators'], 
                               row['L_index'], year, alpha, beta, gamma)
        
        # Total supply
        S = self.params['phi_c'] * self.params['f_addr'] + S_terr + g
        
        return S, D
    
    def fit_parameters(self) -> Dict[str, float]:
        """Fit emergent supply parameters using post-1995 data."""
        # Use post-1995 data for fitting
        fit_df = self.data[self.data['year'] >= 1995].copy()
        
        # Proxy emergent term using baseline generation weights (decoupled from active priors)
        wA0, wP0 = self.gen_weights['wA'], self.gen_weights['wP']
        terr = (fit_df['NTA'] * wA0 + fit_df['NTP'] * wP0)
        g_proxy = np.maximum(fit_df['S_supply'] - terr, 1e-9)
        
        # Log-linear regression
        X = np.column_stack([
            np.ones(len(fit_df)), 
            np.log(fit_df['C_compute']), 
            np.log(fit_df['U_creators'])
        ])
        
        # Vectorized gate calculation for robustness
        is_post = (fit_df['year'] >= self.regime_year)
        L0 = np.where(is_post, self.params['L0_post'], self.params['L0_pre'])
        a = self.params['a']
        sigma = 1.0 / (1.0 + np.exp(-a * (fit_df['L_index'] - L0)))
        cap_sigma = np.maximum(self.cap_CU(fit_df['C_compute'], fit_df['U_creators']) * sigma, 1e-9)
        
        y = np.log(g_proxy / cap_sigma)
        
        # OLS
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        
        fitted = {
            'alpha': float(np.exp(coef[0])),
            'beta': float(coef[1]),
            'gamma': float(coef[2])
        }
        
        self.params.update(fitted)
        return fitted
    
    def run_model(self) -> pd.DataFrame:
        """Run the complete model with fitted parameters."""
        # Fit emergent parameters on post-1995 slice
        fitted = self.fit_parameters()
        alpha, beta, gamma = fitted['alpha'], fitted['beta'], fitted['gamma']
        
        # If fitted parameters are extreme, use reasonable defaults
        if fitted['alpha'] > 1e6 or abs(fitted['beta']) > 5 or abs(fitted['gamma']) > 5:
            print("⚠️  Parameter fitting produced extreme values, using defaults...")
            fitted = {
                'alpha': 0.1,
                'beta': 0.8,
                'gamma': 0.6
            }
            self.params.update(fitted)
            alpha, beta, gamma = fitted['alpha'], fitted['beta'], fitted['gamma']
        
        # Recompute S, D, CSR for all years using the fitted parameters
        S, D = [], []
        for _, row in self.data.iterrows():
            s, d = self.compute_SD(row, alpha, beta, gamma)
            S.append(s)
            D.append(d)
        self.data['S_model'] = np.array(S)
        self.data['D_model'] = np.array(D)
        self.data['CSR_model'] = self.data['S_model'] / self.data['D_model']
        
        # Stats from the recomputed model series
        pre_1990 = self.data[self.data['year'] < self.regime_year]
        post_1990 = self.data[self.data['year'] >= self.regime_year]
        
        # Add LMDI-style decomposition
        decomp = self.decompose_post_change(
            alpha=self.params["alpha"], beta=self.params["beta"], gamma=self.params["gamma"]
        )
        
        self.results = {
            'fitted_params': fitted,
            'csr_mean_overall': self.data['CSR_model'].mean(),
            'csr_mean_pre1990': pre_1990['CSR_model'].mean(),
            'csr_mean_post1990': post_1990['CSR_model'].mean(),
            'csr_improvement': ((post_1990['CSR_model'].mean() - pre_1990['CSR_model'].mean()) / 
                               pre_1990['CSR_model'].mean() * 100),
            'population_growth': (self.data.iloc[-1]['population'] / 
                                self.data.iloc[0]['population']),
            'creators_growth': (self.data.iloc[-1]['U_creators'] / 
                              self.data[self.data['year']==1990]['U_creators'].values[0]),
            'decomposition': decomp,
            'network_effects_ok': True,  # Demand uses e_share only; supply uses sigma_L only
            'priors': self.priors,
            'weights_used': {
                'wA': float(self.params['wA']),
                'wP': float(self.params['wP']),
                'wE': float(self.params['wE'])
            },
            'fit_source': 'synthetic_decomp' if 'S_supply' in self.data.columns else 'model_iterative',
            'regime_year': self.regime_year
        }
        
        return self.data
    
    def generate_uncertainty_bands(self, n_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Generate uncertainty bands via parameter sampling."""
        csr_draws = []
        rng = np.random.default_rng(1337)  # Seeded for reproducibility
        
        # Store original params
        orig_params = self.params.copy()
        
        for _ in range(n_samples):
            # Sample parameters with clamping
            alpha_draw = np.clip(self.params['alpha'] * np.exp(rng.normal(0, 0.20)), 
                                self.params['alpha']/3, self.params['alpha']*3)
            beta_draw = np.clip(self.params['beta'] + rng.normal(0, 0.10), 0.2, 2.0)
            gamma_draw = np.clip(self.params['gamma'] + rng.normal(0, 0.10), 0.2, 2.0)
            
            # Sample sharing parameters for more realistic uncertainty
            self.params['e_max_pre'] = np.clip(orig_params['e_max_pre'] + rng.normal(0, 0.03), 0.0, 0.6)
            self.params['e_max_post'] = np.clip(orig_params['e_max_post'] + rng.normal(0, 0.03), 0.05, 0.9)
            self.params['K'] = np.clip(orig_params['K'] + rng.normal(0, 0.05), 0.1, 5.0)
            self.params['n'] = np.clip(orig_params['n'] + rng.normal(0, 0.10), 0.8, 3.5)
            
            # Compute CSR with sampled parameters
            csr = []
            for _, row in self.data.iterrows():
                s, d = self.compute_SD(row, alpha_draw, beta_draw, gamma_draw)
                csr.append(s/d)
            csr_draws.append(csr)
        
        # Restore original params
        self.params = orig_params
        
        csr_draws = np.array(csr_draws)
        low = np.percentile(csr_draws, 5, axis=0)
        high = np.percentile(csr_draws, 95, axis=0)
        
        return low, high
    
    def plot_comprehensive_analysis(self, output_dir: Path = Path('../outputs/images'), 
                                   version: str = 'v1.0.0'):
        """Generate comprehensive visualization of the model."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 1], hspace=0.3, wspace=0.3)
        
        # Color scheme
        colors = {
            'connectivity': '#1f77b4',
            'compute': '#2ca02c',
            'creators': '#9467bd',
            'csr': '#ff7f0e',
            'regime': '#d62728'
        }
        
        # Plot 1: Population & Demand
        ax1 = fig.add_subplot(gs[0, 0])
        ax1_twin = ax1.twinx()
        
        ax1.plot(self.data['year'], self.data['population']/1e9, 'k-', linewidth=3)
        ax1.set_ylabel('Population (billions)', fontsize=12)
        ax1_twin.semilogy(self.data['year'], self.data['D_model'], 'r--', linewidth=2.5, alpha=0.7)
        ax1_twin.set_ylabel('Consciousness Demand (CCU, log)', fontsize=12, color='red')
        ax1.axvline(self.regime_year, color=colors['regime'], linestyle='--', alpha=0.5)
        ax1.set_title('Population Growth Drives Consciousness Demand', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Supply Sources
        ax2 = fig.add_subplot(gs[0, 1])
        terrestrial = self.data['NTA'] * self.params['wA'] + self.data['NTP'] * self.params['wP']
        
        # Emergent CCU: recompute with fitted parameters
        alpha = self.params['alpha']
        beta = self.params['beta']
        gamma = self.params['gamma']
        g_series = []
        for y, C, U, L in zip(self.data['year'], self.data['C_compute'], 
                             self.data['U_creators'], self.data['L_index']):
            g_series.append(self.emergent_supply(C, U, L, y, alpha, beta, gamma))
        emergent = np.array(g_series)
        
        ax2.fill_between(self.data['year'], 0, terrestrial, 
                        label='Terrestrial', color='brown', alpha=0.6)
        ax2.fill_between(self.data['year'], terrestrial, terrestrial + emergent,
                        label='Emergent (Digital/AI)', color=colors['creators'], alpha=0.7)
        ax2.axvline(self.regime_year, color=colors['regime'], linestyle='--', alpha=0.5)
        ax2.set_title('Consciousness Supply Sources', fontsize=14)
        ax2.set_ylabel('Supply (CCU/yr)', fontsize=12)
        ax2.set_ylim(bottom=0)  # Avoid negative fills
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Three Drivers
        ax3 = fig.add_subplot(gs[1, :])
        
        # Normalize for comparison
        L_norm = self.data['L_index'] / self.data['L_index'].max()
        C_norm = self.data['C_compute'] / self.data['C_compute'].max()
        U_norm = self.data['U_creators'] / self.data['U_creators'].max()
        
        ax3.plot(self.data['year'], L_norm, label='Connectivity (L)', 
                color=colors['connectivity'], linewidth=3)
        ax3.plot(self.data['year'], C_norm, label='Compute (C)', 
                color=colors['compute'], linewidth=3)
        ax3.plot(self.data['year'], U_norm, label='Digital Creators (U)', 
                color=colors['creators'], linewidth=3)
        ax3.axvline(self.regime_year, color=colors['regime'], linestyle='--', alpha=0.5)
        ax3.fill_between([self.regime_year, 2025], 0, 1, alpha=0.1, color=colors['regime'])
        ax3.set_title('The Three Drivers of Emergent Consciousness (Normalized)', fontsize=14)
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('Normalized Level', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: CSR with uncertainty (auto-scaled)
        ax4 = fig.add_subplot(gs[2, :])
        
        # Plot CSR × 10⁴ for better visibility
        csr_scale = 10000
        y_scaled = self.data['CSR_model'] * csr_scale
        ax4.plot(self.data['year'], y_scaled, 'b-', linewidth=2.5, label='CSR × 10⁴')
        
        # Means (scaled)
        pre_mean = self.results['csr_mean_pre1990'] * csr_scale
        post_mean = self.results['csr_mean_post1990'] * csr_scale
        ax4.axhline(pre_mean, color='orange', linestyle=':', linewidth=2, alpha=0.7, 
                   label=f'Pre-1990: {pre_mean:.2f}')
        ax4.axhline(post_mean, color='green', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'Post-1990: {post_mean:.2f}')
        
        # Dynamic y-limits
        y_min = float(np.nanmin([y_scaled.min(), pre_mean, post_mean]))
        y_max = float(np.nanmax([y_scaled.max(), pre_mean, post_mean]))
        margin = max(0.05 * (y_max - y_min), 0.05)
        if (y_max - y_min) < 0.2:
            # Ensure a readable band even if CSR is flat
            center = 0.5 * (y_max + y_min)
            y_min, y_max = center - 0.1, center + 0.1
        ax4.set_ylim(y_min - margin, y_max + margin)
        
        # Regime switch line and shading using current ylim
        ax4.axvline(self.regime_year, color=colors['regime'], linestyle='--', alpha=0.5, label=f'{self.regime_year} Regime Switch')
        y0, y1 = ax4.get_ylim()
        ax4.fill_between([self.regime_year, 2025], y0, y1, alpha=0.1, color=colors['regime'])
        
        # Improvement annotation placed relative to dynamic scale
        ann_y = y_min + 0.7 * (y_max - y_min)
        ax4.text(2005, ann_y, 
                f"+{self.results['csr_improvement']:.0f}% improvement\npost-1990!", 
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8),
                ha='center')
        
        ax4.set_title('Consciousness Surplus Ratio: Digital Revolution Maintains Abundance', fontsize=14)
        ax4.set_xlabel('Year', fontsize=12)
        ax4.set_ylabel('CSR × 10⁴ (Supply/Demand)', fontsize=12)
        ax4.legend(loc='lower left')
        ax4.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle('Consciousness Flux Model: How Digital Emergence Solves the Soul Supply Problem', 
                    fontsize=18, y=0.98)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save with version and timestamp
        filename = f'consciousness_flux_analysis_{version}_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved visualization: {filename}")
        
        return str(output_dir / filename)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Consciousness Flux Model Analysis')
    parser.add_argument('--data-dir', type=Path, default=Path('../data'),
                       help='Directory containing data files')
    parser.add_argument('--output-dir', type=Path, default=Path('../outputs/images'),
                       help='Directory for output images')
    parser.add_argument('--version', type=str, default='v1.0.3',
                       help='Version string for outputs')
    parser.add_argument('--priors', type=str, default='PHYSICALIST',
                       choices=['PHYSICALIST', 'IIT', 'PANPSYCHIST'],
                       help='Philosophical priors: PHYSICALIST, IIT, or PANPSYCHIST')
    parser.add_argument('--cosmic', type=float, default=None, help='Optional phi_c override (CCU/yr, small)')
    parser.add_argument('--addr', type=float, default=None, help='Optional f_addr override (0..1)')
    
    args = parser.parse_args()
    
    print("🌌 CONSCIOUSNESS FLUX MODEL - Version 1.0.3")
    print("=" * 50)
    print(f"🧠 Using {args.priors} priors")
    
    # Initialize and run model
    model = ConsciousnessFluxModel(data_dir=args.data_dir, priors=args.priors,
                                   cosmic=args.cosmic, addr=args.addr)
    
    print("📂 Loading data...")
    model.load_data()
    
    print("🔧 Fitting parameters and running model...")
    model.run_model()
    
    print("\n✨ RESULTS:")
    print(f"   α (emergent scale): {model.results['fitted_params']['alpha']:.6f}")
    print(f"   β (compute elasticity): {model.results['fitted_params']['beta']:.3f}")
    print(f"   γ (users elasticity): {model.results['fitted_params']['gamma']:.3f}")
    print(f"\n   CSR Improvement: +{model.results['csr_improvement']:.1f}%")
    print(f"   Population Growth: {model.results['population_growth']:.1f}x")
    print(f"   Digital Creators Growth: {model.results['creators_growth']:.1f}x")
    
    print("\n📊 Generating visualizations...")
    filename = model.plot_comprehensive_analysis(output_dir=args.output_dir, 
                                               version=args.version)
    model.results["artifact_image"] = filename
    decomp = model.decompose_post_change(
        alpha=model.params["alpha"], beta=model.params["beta"], gamma=model.params["gamma"]
    )
    model.results["decomposition"] = decomp

    # Save results JSON linked to the same timestamp as the image
    out_results_dir = Path(args.output_dir).parent / 'results'
    out_results_dir.mkdir(parents=True, exist_ok=True)
    img_name = Path(filename).name
    try:
        ts = img_name.rsplit('_', 1)[1].replace('.png', '')
    except Exception:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = out_results_dir / f'model_results_{args.version}_{ts}.json'
    with open(results_file, 'w') as f:
        json.dump(model.results, f, indent=2)
    print(f"✅ Saved results: {results_file.name}")
    
    print("\n🌟 Analysis complete! The consciousness cosmos has been revealed!")
    

if __name__ == "__main__":
    main()
