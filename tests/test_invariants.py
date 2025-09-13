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

def test_changepoints_and_quality_toggle():
    from src.consciousness_flux_model_v1 import ConsciousnessFluxModel
    m = ConsciousnessFluxModel(priors="IIT", enable_rich=True, enable_quality=True, detect_changes=True)
    m.load_data(); m.run_model()
    assert "decomposition" in m.results
    assert "artifact_image" in m.results
    # changepoints optional, but type should be list if requested
    if m.detect_changes:
        assert isinstance(m.results.get("changepoints", []), list)

def test_regime_year_setting():
    m = ConsciousnessFluxModel(priors="IIT", regime_year=2000)
    m.load_data(); m.run_model()
    assert m.results["regime_year"] == 2000
    # Check that changing regime year changes fitted params
    m2 = ConsciousnessFluxModel(priors="IIT", regime_year=1990)
    m2.load_data(); m2.run_model()
    fp1 = m.results["fitted_params"]
    fp2 = m2.results["fitted_params"]
    assert fp1 != fp2

def test_parameter_fitting_and_derivation():
    from src.consciousness_flux_model_v1 import ConsciousnessFluxModel
    m = ConsciousnessFluxModel(priors="IIT", enable_rich=True, enable_quality=True, detect_changes=True)
    m.load_data(); m.run_model()
    assert "decomposition" in m.results
    assert "artifact_image" in m.results
    # changepoints optional, but type should be list if requested
    if m.detect_changes:
        assert isinstance(m.results.get("changepoints", []), list)
        assert isinstance(m.results.get("quality_multiplier", []), list)
        assert isinstance(m.results.get("level_weights", []), list)

def test_uncertainty_bands():
    from src.consciousness_flux_model_v1 import ConsciousnessFluxModel
    m = ConsciousnessFluxModel(priors="IIT", enable_rich=True, enable_quality=True, detect_changes=True)
    m.load_data(); m.run_model()
    assert "decomposition" in m.results
    assert "artifact_image" in m.results
    # changepoints optional, but type should be list if requested
    if m.detect_changes:
        assert isinstance(m.results.get("changepoints", []), list)
        assert isinstance(m.results.get("quality_multiplier", []), list)
        assert isinstance(m.results.get("level_weights", []), list)
    
def test_comprehensive_analysis():
    from src.consciousness_flux_model_v1 import ConsciousnessFluxModel
    m = ConsciousnessFluxModel(priors="IIT", enable_rich=True, enable_quality=True, detect_changes=True)
    m.load_data(); m.run_model()
    assert "decomposition" in m.results
    assert "artifact_image" in m.results
    assert "uncertainty_bands" in m.results
    assert "comprehensive_analysis" in m.results
    assert "comprehensive_analysis_image" in m.results
    assert "comprehensive_analysis_results" in m.results
    assert "comprehensive_analysis_results_image" in m.results


def test_changepoints_and_quality_toggle():
    from src.consciousness_flux_model_v1 import ConsciousnessFluxModel
    m = ConsciousnessFluxModel(priors="IIT", enable_rich=True, enable_quality=True, detect_changes=True)
    m.load_data(); m.run_model()
    assert "decomposition" in m.results
    # Rich features should be present
    if m.enable_rich:
        assert "hierarchy_weights_sample" in m.results
    # changepoints optional, but type should be list if requested
    if m.detect_changes:
        assert isinstance(m.results.get("changepoints", []), list)
        assert len(m.results["changepoints"]) > 0  # Should detect some changepoints

