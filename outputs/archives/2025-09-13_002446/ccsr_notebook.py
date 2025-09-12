#!/usr/bin/env python3
"""
Consciousness Flux Model (RFE) - Reincarnation Flux Equation
============================================================

A mathematical framework modeling consciousness as energy flowing through
cosmic, terrestrial, and emergent sources to meet human demand.

Author: Consciousness Flux Research Team
Date: 2025-09-12
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.stats import gaussian_kde
import warnings
from typing import Tuple, Dict, Optional

class ConsciousnessFluxModel:
    """
    Implementation of the Reincarnation Flux Equation (RFE).
    
    Models consciousness supply from:
    - Cosmic flux (souls from beyond)
    - Terrestrial recycling (animal/plant consciousness)
    - Emergent creation (digital/AI consciousness)
    
    Meeting human demand through population and efficiency dynamics.
    """
    
    def __init__(self, params_file: str = 'ccsr_parameters_template.csv'):
        """Initialize model with parameter template."""
        self.params_df = pd.read_csv(params_file)
        self.params = self._extract_default_params()
        self.fitted_params = None
        self.data = None
        
    def _extract_default_params(self) -> Dict[str, float]:
        """Extract default parameter values from template."""
        params = {}
        for _, row in self.params_df.iterrows():
            if pd.notna(row['default']):
                params[row['symbol']] = float(row['default'])
        return params
    
    def load_data(self, data_file: str = 'synthetic_ccsr_timeseries.csv') -> pd.DataFrame:
        """Load time series data."""
        self.data = pd.read_csv(data_file)
        return self.data
    
    def hill_function(self, L: np.ndarray, e_max: float, K: float, n: float) -> np.ndarray:
        """Hill function for efficiency calculation."""
        return e_max * (L**n) / (K**n + L**n)
    
    def logistic_gate(self, L: np.ndarray, a: float, L0: float) -> np.ndarray:
        """Logistic gate function for connectivity-dependent processes."""
        return 1 / (1 + np.exp(-a * (L - L0)))
    
    def emergent_creation(self, C: np.ndarray, U: np.ndarray, L: np.ndarray,
                         alpha: float, beta: float, gamma: float,
                         Csat: float, Usat: float, p: float, q: float,
                         a: float, L0: float) -> np.ndarray:
        """Calculate emergent consciousness creation."""
        # Power law with saturation
        numerator = alpha * (C**beta) * (U**gamma)
        denominator = 1 + (C/Csat)**p + (U/Usat)**q
        
        # Apply connectivity gate
        gate = self.logistic_gate(L, a, L0)
        
        return numerator / denominator * gate
    
    def calculate_demand(self, P: np.ndarray, L: np.ndarray, year: np.ndarray) -> np.ndarray:
        """Calculate consciousness demand including efficiency gains."""
        # Determine regime-specific parameters
        is_post_1990 = year >= 1990
        e_max = np.where(is_post_1990, self.params['e_max_post'], self.params['e_max_pre'])
        
        # Calculate efficiency
        efficiency = self.hill_function(L, e_max, self.params['K'], self.params['n'])
        
        # Base demand with efficiency gains
        cH_dot = self.params['cH_dot']
        demand = P * cH_dot * (1 - efficiency)
        
        return demand
    
    def calculate_supply(self, C: np.ndarray, U: np.ndarray, L: np.ndarray,
                        NTA: np.ndarray, NTP: np.ndarray, year: np.ndarray,
                        alpha: float, beta: float, gamma: float) -> np.ndarray:
        """Calculate total consciousness supply."""
        # Terrestrial recycling
        wA = self.params['wA']
        wP = self.params['wP']
        terrestrial = wA * NTA + wP * NTP
        
        # Emergent creation with regime switching
        is_post_1990 = year >= 1990
        L0 = np.where(is_post_1990, self.params['L0_post'], self.params['L0_pre'])
        
        emergent = self.emergent_creation(
            C, U, L, alpha, beta, gamma,
            self.params['Csat'], self.params['Usat'],
            self.params['p'], self.params['q'],
            self.params['a'], L0
        )
        
        # Cosmic flux (optional, set to 0 in baseline)
        cosmic = self.params['phi_c'] * self.params['f_addr']
        
        return terrestrial + emergent + cosmic
    
    def objective_function(self, params: np.ndarray, data: pd.DataFrame) -> float:
        """Objective function for parameter fitting."""
        alpha, beta, gamma = params
        
        # Calculate predicted supply
        supply_pred = self.calculate_supply(
            data['C_compute'].values,
            data['U_creators'].values,
            data['L_index'].values,
            data['NTA'].values,
            data['NTP'].values,
            data['year'].values,
            alpha, beta, gamma
        )
        
        # Calculate predicted demand
        demand_pred = self.calculate_demand(
            data['population'].values,
            data['L_index'].values,
            data['year'].values
        )
        
        # Calculate predicted CSR
        csr_pred = supply_pred / demand_pred
        
        # Log-space MSE for CSR
        log_csr_actual = np.log(data['CSR'].values)
        log_csr_pred = np.log(csr_pred)
        
        mse = np.mean((log_csr_actual - log_csr_pred)**2)
        
        return mse
    
    def fit_parameters(self, method: str = 'minimize') -> Dict[str, float]:
        """Fit model parameters to data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Initial guesses from template
        x0 = [self.params['alpha'], self.params['beta'], self.params['gamma']]
        
        # Bounds from template
        bounds = [
            (self.params_df[self.params_df['symbol'] == 'alpha']['prior_lower'].values[0],
             self.params_df[self.params_df['symbol'] == 'alpha']['prior_upper'].values[0]),
            (self.params_df[self.params_df['symbol'] == 'beta']['prior_lower'].values[0],
             self.params_df[self.params_df['symbol'] == 'beta']['prior_upper'].values[0]),
            (self.params_df[self.params_df['symbol'] == 'gamma']['prior_lower'].values[0],
             self.params_df[self.params_df['symbol'] == 'gamma']['prior_upper'].values[0])
        ]
        
        # Optimize
        result = minimize(
            lambda x: self.objective_function(x, self.data),
            x0, bounds=bounds, method='L-BFGS-B'
        )
        
        if result.success:
            self.fitted_params = {
                'alpha': result.x[0],
                'beta': result.x[1],
                'gamma': result.x[2]
            }
            print(f"✨ Model fitted successfully!")
            print(f"   α (emergent scale): {self.fitted_params['alpha']:.6f}")
            print(f"   β (compute elasticity): {self.fitted_params['beta']:.3f}")
            print(f"   γ (users elasticity): {self.fitted_params['gamma']:.3f}")
        else:
            warnings.warn("Optimization failed. Using default parameters.")
            self.fitted_params = {
                'alpha': self.params['alpha'],
                'beta': self.params['beta'],
                'gamma': self.params['gamma']
            }
        
        return self.fitted_params
    
    def generate_predictions(self, include_uncertainty: bool = True) -> pd.DataFrame:
        """Generate model predictions with uncertainty bands."""
        if self.fitted_params is None:
            self.fit_parameters()
        
        # Calculate predictions
        results = self.data.copy()
        
        # Use fitted parameters
        alpha = self.fitted_params['alpha']
        beta = self.fitted_params['beta']
        gamma = self.fitted_params['gamma']
        
        # Calculate supply and demand
        results['supply_pred'] = self.calculate_supply(
            results['C_compute'].values,
            results['U_creators'].values,
            results['L_index'].values,
            results['NTA'].values,
            results['NTP'].values,
            results['year'].values,
            alpha, beta, gamma
        )
        
        results['demand_pred'] = self.calculate_demand(
            results['population'].values,
            results['L_index'].values,
            results['year'].values
        )
        
        results['csr_pred'] = results['supply_pred'] / results['demand_pred']
        
        if include_uncertainty:
            # Simple uncertainty estimation (±10% for demonstration)
            # In production, use proper Bayesian inference or bootstrap
            results['csr_lower'] = results['csr_pred'] * 0.9
            results['csr_upper'] = results['csr_pred'] * 1.1
        
        return results
    
    def plot_drivers(self, save_path: Optional[str] = None):
        """Plot the three main drivers: L, C, U over time."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Connectivity Index (L)
        ax1.plot(self.data['year'], self.data['L_index'], 'b-', linewidth=2, label='L (Connectivity)')
        ax1.axvline(x=1990, color='red', linestyle='--', alpha=0.7, label='1990 Regime Switch')
        ax1.fill_between([1990, self.data['year'].max()], 0, ax1.get_ylim()[1], 
                        alpha=0.1, color='red', label='Post-1990 Era')
        ax1.set_ylabel('Connectivity Index', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Compute (C)
        ax2.plot(self.data['year'], self.data['C_compute'], 'g-', linewidth=2, label='C (Compute Power)')
        ax2.axvline(x=1990, color='red', linestyle='--', alpha=0.7)
        ax2.fill_between([1990, self.data['year'].max()], 0, ax2.get_ylim()[1], 
                        alpha=0.1, color='red')
        ax2.set_ylabel('Compute Units', fontsize=12)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Creators/Users (U)
        ax3.plot(self.data['year'], self.data['U_creators']/1e6, 'm-', linewidth=2, label='U (Digital Creators)')
        ax3.axvline(x=1990, color='red', linestyle='--', alpha=0.7)
        ax3.fill_between([1990, self.data['year'].max()], 0, ax3.get_ylim()[1], 
                        alpha=0.1, color='red')
        ax3.set_ylabel('Millions of Creators', fontsize=12)
        ax3.set_xlabel('Year', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle('🌟 Consciousness Flux Model: Key Drivers Evolution', fontsize=16, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_csr_analysis(self, save_path: Optional[str] = None):
        """Plot CSR with uncertainty bands and key statistics."""
        results = self.generate_predictions()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot actual CSR
        ax.plot(results['year'], results['CSR']*10000, 'ko-', markersize=4, 
                linewidth=1, alpha=0.6, label='Observed CSR (×10⁴)')
        
        # Plot predicted CSR with uncertainty
        ax.plot(results['year'], results['csr_pred']*10000, 'b-', linewidth=2.5, 
                label='Model Prediction')
        ax.fill_between(results['year'], results['csr_lower']*10000, 
                       results['csr_upper']*10000, alpha=0.2, color='blue', 
                       label='Uncertainty Band (±10%)')
        
        # Mark 1990 regime switch
        ax.axvline(x=1990, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                  label='1990 Regime Switch')
        ax.fill_between([1990, results['year'].max()], 0, ax.get_ylim()[1], 
                       alpha=0.1, color='red')
        
        # Add horizontal line at CSR = 1
        ax.axhline(y=1, color='green', linestyle=':', linewidth=2, alpha=0.7, 
                  label='CSR = 1 (Perfect Balance)')
        
        # Annotations
        pre_1990_mean = results[results['year'] < 1990]['CSR'].mean() * 10000
        post_1990_mean = results[results['year'] >= 1990]['CSR'].mean() * 10000
        
        ax.text(1970, 1.3, f'Pre-1990 Mean: {pre_1990_mean:.2f}×10⁻⁴', 
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        ax.text(2000, 1.3, f'Post-1990 Mean: {post_1990_mean:.2f}×10⁻⁴', 
                fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
        
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Consciousness Surplus Ratio (×10⁴)', fontsize=14)
        ax.set_title('🌌 Consciousness Surplus Ratio: Abundance Across Decades', fontsize=16)
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to show abundance clearly
        ax.set_ylim(0.8, 1.4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary_statistics(self):
        """Print comprehensive model statistics."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        results = self.generate_predictions()
        
        print("\n" + "="*60)
        print("🌟 CONSCIOUSNESS FLUX MODEL - SUMMARY STATISTICS 🌟")
        print("="*60)
        
        # Model parameters
        print("\n📊 FITTED PARAMETERS:")
        print(f"   α (emergent scale): {self.fitted_params['alpha']:.6f} CCU/yr")
        print(f"   β (compute elasticity): {self.fitted_params['beta']:.3f}")
        print(f"   γ (users elasticity): {self.fitted_params['gamma']:.3f}")
        
        # CSR statistics
        print("\n📈 CONSCIOUSNESS SURPLUS RATIO (CSR):")
        print(f"   Overall Mean: {results['CSR'].mean():.6f}")
        print(f"   Pre-1990 Mean: {results[results['year'] < 1990]['CSR'].mean():.6f}")
        print(f"   Post-1990 Mean: {results[results['year'] >= 1990]['CSR'].mean():.6f}")
        print(f"   Minimum: {results['CSR'].min():.6f} (Year {results.loc[results['CSR'].idxmin(), 'year']:.0f})")
        print(f"   Maximum: {results['CSR'].max():.6f} (Year {results.loc[results['CSR'].idxmax(), 'year']:.0f})")
        
        # Supply composition
        print("\n⚡ SUPPLY COMPOSITION (2025):")
        latest = results[results['year'] == 2025].iloc[0]
        terrestrial = self.params['wA'] * latest['NTA'] + self.params['wP'] * latest['NTP']
        emergent = latest['Gs']
        total_supply = latest['supply_pred']
        
        print(f"   Terrestrial (Animals + Plants): {terrestrial:.0f} CCU ({terrestrial/total_supply*100:.1f}%)")
        print(f"   Emergent (Digital/AI): {emergent:.0f} CCU ({emergent/total_supply*100:.1f}%)")
        print(f"   Total Supply: {total_supply:.0f} CCU")
        
        # Demand statistics
        print("\n🌍 DEMAND STATISTICS (2025):")
        print(f"   Population: {latest['population']/1e9:.2f} billion")
        print(f"   Total Demand: {latest['demand_pred']:.0f} CCU")
        print(f"   Per-capita (with efficiency): {latest['demand_pred']/latest['population']:.6f} CCU/person")
        
        # Growth rates
        print("\n📊 GROWTH RATES (1950-2025):")
        pop_growth = (results['population'].iloc[-1] / results['population'].iloc[0]) ** (1/75) - 1
        compute_growth = (results['C_compute'].iloc[-1] / results['C_compute'].iloc[0]) ** (1/75) - 1
        users_growth = (results['U_creators'].iloc[-1] / results['U_creators'].iloc[0]) ** (1/75) - 1
        
        print(f"   Population CAGR: {pop_growth*100:.2f}%")
        print(f"   Compute CAGR: {compute_growth*100:.2f}%")
        print(f"   Digital Creators CAGR: {users_growth*100:.2f}%")
        
        print("\n" + "="*60)
        print("✨ The cosmos maintains consciousness abundance! ✨")
        print("="*60 + "\n")


def main():
    """Main execution function."""
    print("🌌 Initializing Consciousness Flux Model...")
    
    # Initialize model
    model = ConsciousnessFluxModel()
    
    # Load data
    print("📂 Loading synthetic time series data...")
    model.load_data()
    
    # Fit parameters
    print("🔧 Fitting model parameters...")
    model.fit_parameters()
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    model.plot_drivers(save_path='consciousness_drivers.png')
    model.plot_csr_analysis(save_path='consciousness_surplus_ratio.png')
    
    # Print summary
    model.print_summary_statistics()
    
    print("\n✅ Analysis complete! The consciousness cosmos has been revealed! 🌟")


if __name__ == "__main__":
    main()
