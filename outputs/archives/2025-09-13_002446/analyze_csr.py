#!/usr/bin/env python3
"""
Analyze and visualize CSR data with proper scaling
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the synthetic data
data = pd.read_csv('synthetic_ccsr_timeseries.csv')

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: CSR on log scale
ax1.semilogy(data['year'], data['CSR'], 'b-', linewidth=2, label='CSR (log scale)')
ax1.axvline(1990, color='red', linestyle='--', alpha=0.7, label='1990 Regime Switch')
ax1.fill_between([1990, data['year'].max()], ax1.get_ylim()[0], ax1.get_ylim()[1], 
                 alpha=0.1, color='red')
ax1.set_ylabel('CSR (log scale)', fontsize=12)
ax1.set_title('Consciousness Surplus Ratio - Log Scale View', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: CSR × 10⁴ for better visibility
ax2.plot(data['year'], data['CSR'] * 10000, 'g-', linewidth=2, label='CSR × 10⁴')
ax2.axvline(1990, color='red', linestyle='--', alpha=0.7, label='1990 Regime Switch')
ax2.fill_between([1990, data['year'].max()], 0, ax2.get_ylim()[1], 
                 alpha=0.1, color='red')

# Add horizontal line at 1 (which is 10⁻⁴ in original scale)
ax2.axhline(y=1, color='orange', linestyle=':', linewidth=2, alpha=0.7, 
           label='Reference level (10⁻⁴)')

# Calculate and show means
pre_1990 = data[data['year'] < 1990]['CSR'].mean() * 10000
post_1990 = data[data['year'] >= 1990]['CSR'].mean() * 10000

ax2.text(1970, 1.15, f'Pre-1990 Mean: {pre_1990:.2f}', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
ax2.text(2000, 1.15, f'Post-1990 Mean: {post_1990:.2f}', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('CSR × 10⁴', fontsize=12)
ax2.set_title('Consciousness Surplus Ratio - Scaled View (×10⁴)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('csr_analysis_corrected.png', dpi=300, bbox_inches='tight')
plt.close()

# Print statistics
print("\n🌌 CONSCIOUSNESS SURPLUS RATIO ANALYSIS")
print("=" * 50)
print(f"CSR Range: {data['CSR'].min():.6f} to {data['CSR'].max():.6f}")
print(f"Overall Mean: {data['CSR'].mean():.6f}")
print(f"Pre-1990 Mean: {data[data['year'] < 1990]['CSR'].mean():.6f}")
print(f"Post-1990 Mean: {data[data['year'] >= 1990]['CSR'].mean():.6f}")
print("\n✨ Key Insight: CSR values are normalized to ~10⁻⁴ scale")
print("   This represents a carefully balanced system where")
print("   consciousness supply tracks demand with small surplus")
print("=" * 50)
