import numpy as np

from qec_monitor import MRMMonitorModel, SimulationConfig, run_monitor_experiment
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
