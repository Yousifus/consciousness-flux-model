# src/rich/visualize_multi.py
import matplotlib.pyplot as plt
import numpy as np

def plot_csr_overlay_by_priors(df, out_path):
    plt.figure(figsize=(9,4))
    for pri in sorted(df["priors"].unique()):
        sub = df[df["priors"]==pri]
        plt.plot(sub["year"], sub["CSR_model"], label=pri)
    plt.axvline(1990, linestyle="--")
    plt.title("CSR by Priors")
    plt.xlabel("Year"); plt.ylabel("S/D")
    plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()
