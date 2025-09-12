from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from consciousness_flux_model_v1 import ConsciousnessFluxModel


def run(priors_list, cosmic=None, addr=None):
    frames = []
    for p in priors_list:
        m = ConsciousnessFluxModel(priors=p, cosmic=cosmic, addr=addr)
        m.load_data()
        m.run_model()
        df = m.data[["year", "CSR_model"]].copy()
        df["priors"] = p
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out_dir = Path("../outputs/results"); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "csr_by_priors.csv"
    out.to_csv(csv_path, index=False)

    # overlay plot (default matplotlib styling)
    plt.figure(figsize=(9, 4))
    for p in priors_list:
        d = out[out["priors"] == p]
        plt.plot(d["year"], d["CSR_model"], label=p)
    plt.axvline(1990, linestyle="--")
    plt.title("CSR by Priors")
    plt.xlabel("Year"); plt.ylabel("S/D")
    plt.legend()
    img_dir = Path("../outputs/images"); img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "csr_by_priors.png"
    plt.tight_layout(); plt.savefig(img_path, dpi=200); plt.close()

    print("Saved:", csv_path, img_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", default="PHYSICALIST,IIT,PANPSYCHIST")
    ap.add_argument("--cosmic", type=float, default=None)
    ap.add_argument("--addr", type=float, default=None)
    args = ap.parse_args()
    priors_list = [s.strip() for s in args.priors.split(",") if s.strip()]
    run(priors_list, cosmic=args.cosmic, addr=args.addr)


