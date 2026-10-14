import numpy as np
import pytest

from qec_monitor import (
    BackendSamples,
    StimMemoryBackend,
    run_backend_experiment,
    run_surface_code_experiment,
)


class _KnownAnswerBackend:
    def __init__(self):
        self.monitor_seen = None

    def sample(self, shots, seed=None):
        return BackendSamples(
            detection_events=np.array([[0, 1], [1, 0]]),
            observable_flips=np.array([[0], [1]]),
            monitor_observations=np.array([[0.1, 0.2], [0.8, 0.9]]),
        )

    def decode(self, detection_events, monitor_observations=None):
        self.monitor_seen = monitor_observations
        return np.array([[0], [1]])


def test_backend_neutral_runner_forwards_monitor_observations_and_scores_logical_bits():
    backend = _KnownAnswerBackend()
    result = run_backend_experiment(backend, shots=2, seed=12)

    assert result["logical_error_rate"] == 0.0
    np.testing.assert_array_equal(backend.monitor_seen, [[0.1, 0.2], [0.8, 0.9]])


def test_stim_backend_returns_separate_detector_and_observable_samples():
    backend = StimMemoryBackend(distance=3, rounds=2, p_phys=0.001)
    result = run_backend_experiment(backend, shots=16, seed=7)
    samples = result["samples"]

    assert samples.detection_events.shape == (16, backend.circuit.num_detectors)
    assert samples.observable_flips.shape == (16, backend.circuit.num_observables)
    assert 0.0 <= result["logical_error_rate"] <= 1.0


def test_surface_code_experiment_reports_observable_error_not_raw_measurement_mean():
    result = run_surface_code_experiment(distance=3, rounds=2, p_phys=0.001, shots=16, seed=7)

    assert result["monitor_used"] is False
    assert len(result["observable_error_rates"]) == 1
    assert result["logical_error_rate"] == pytest.approx(result["observable_error_rates"][0])


def test_backend_sample_arrays_require_matching_shots_and_binary_quantum_bits():
    with pytest.raises(ValueError, match="same shots"):
        BackendSamples(np.zeros((2, 3)), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="binary"):
        BackendSamples(np.array([[0, 2]]), np.zeros((1, 1)))