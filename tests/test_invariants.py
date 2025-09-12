import os, sys
import numpy as np

# Ensure project root is on sys.path so 'src' is importable when running this file directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.consciousness_flux_model_v1 import ConsciousnessFluxModel


def test_positivity_and_network_effects():
    m = ConsciousnessFluxModel(priors="IIT")
    m.load_data(); m.run_model()
    assert (m.data["S_model"] > 0).all() and (m.data["D_model"] > 0).all()
    assert (m.data["CSR_model"] >= 0).all()
    assert m.results["network_effects_ok"] is True


def test_priors_independence_on_synthetic():
    vals = []
    for p in ("PHYSICALIST", "IIT", "PANPSYCHIST"):
        m = ConsciousnessFluxModel(priors=p); m.load_data(); m.run_model()
        fp = m.results["fitted_params"]
        vals.append((round(fp["alpha"], 3), round(fp["beta"], 2), round(fp["gamma"], 2)))
    # synthetic file was generated under one set of weights, so fits should not drift with priors
    assert len(set(vals)) == 1


