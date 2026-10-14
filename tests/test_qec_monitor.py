import numpy as np
import pytest

from qec_monitor import (
    MRMMonitorModel,
    SimulationConfig,
    run_exact_surface_code_experiment,
    run_monitor_experiment,
)
from qec_monitor.decoder import match_and_decode


def test_monitor_estimates_signal_metrics():
    model = MRMMonitorModel(input_amplitude=1.0, sigma=0.8)
    t = np.linspace(-3, 3, 256)
    signal = model.drop_port_signal(t, amplitude_error=0.1, phase_error=0.2)
    metrics = model.estimate_control_error(signal, t)
    assert metrics.amplitude > 0.0
    assert metrics.pulse_area > 0.0
    assert 0.0 <= model.accuracy_score(metrics) <= 1.0


def test_match_and_decode_returns_valid_correction_mask():
    syndrome = [1, 4, 7]
    decoded = match_and_decode(syndrome, n_data=9, base_error_rate=0.01, monitor_quality=0.8)
    assert decoded.shape == (9,)
    assert decoded.dtype.kind in {"i", "u", "b"}


def test_run_monitor_experiment_returns_finite_rates():
    result = run_monitor_experiment(SimulationConfig(distance=3, shots=20, monitor_quality=0.8))
    assert set(result) >= {"distance", "p_phys", "baseline_logical_error_rate", "monitor_aware_logical_error_rate", "improvement_factor"}
    assert np.isfinite(result["baseline_logical_error_rate"])
    assert np.isfinite(result["monitor_aware_logical_error_rate"])
    assert np.isfinite(result["improvement_factor"])


def test_exact_surface_code_matches_matlab_reference_results():
    default = run_exact_surface_code_experiment()
    stronger_anomaly = run_exact_surface_code_experiment({"p_bad": 0.10})

    assert default["baseline_ler"] == pytest.approx(0.0002113845683, rel=1e-9)
    assert default["monitor_ler"] == pytest.approx(0.0002099323731, rel=1e-9)
    assert stronger_anomaly["baseline_ler"] == pytest.approx(0.0006226238870, rel=1e-9)
    assert stronger_anomaly["monitor_ler"] == pytest.approx(0.0004820882069, rel=1e-9)


def test_exact_surface_code_decoder_controls_and_parameter_validation():
    result = run_exact_surface_code_experiment()

    assert result["perfect_risk_monitor_ler"] <= result["monitor_ler"]
    assert result["monitor_ler"] <= result["baseline_ler"]
    assert result["uninformative_monitor_ler"] == pytest.approx(result["baseline_ler"])
    assert result["shuffled_monitor_ler"] >= result["baseline_ler"]
    assert result["all_builtin_checks_passed"]
    with pytest.raises(ValueError, match="unknown parameter"):
        run_exact_surface_code_experiment({"unknown": 0.1})
    with pytest.raises(ValueError, match="p_bad"):
        run_exact_surface_code_experiment({"p_bad": 0.5})
