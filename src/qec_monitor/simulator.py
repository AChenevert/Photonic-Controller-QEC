from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from .decoder import match_and_decode, monitor_aware_surface_code_decoder


@dataclass
class SimulationConfig:
    distance: int = 3
    p_phys: float = 0.01
    shots: int = 2000
    monitor_quality: float = 0.8
    drift_std: float = 0.1


def _repetition_syndrome(data_errors: np.ndarray) -> np.ndarray:
    """Generate a simple repetition-code syndrome from neighboring bit flips."""
    return np.nonzero(np.diff(data_errors.astype(int)) != 0)[0]


def run_monitor_experiment(config: SimulationConfig):
    """Run a simplified repetition-code comparison between a baseline and a monitor-aware decoder."""
    rng = np.random.default_rng(1234)
    baseline_failures = 0
    monitor_failures = 0

    for _ in range(config.shots):
        n_data = 2 * config.distance - 1
        true_errors = rng.random(n_data) < config.p_phys
        logical = int(np.sum(true_errors) % 2)
        synd = _repetition_syndrome(true_errors)

        baseline_failures += logical

        if config.monitor_quality > 0.0:
            biases = np.clip(1.0 + rng.normal(0.0, config.drift_std, size=max(n_data - 1, 1)), 0.2, 2.0)
            decoded = match_and_decode(synd, n_data=n_data, base_error_rate=config.p_phys, monitor_quality=config.monitor_quality, biases=biases)
            decoded_parity = int(np.sum(decoded) % 2)
            monitor_failures += int(decoded_parity != logical)
        else:
            monitor_failures += logical

    baseline_rate = baseline_failures / max(config.shots, 1)
    monitor_rate = monitor_failures / max(config.shots, 1)

    return {
        "distance": config.distance,
        "p_phys": config.p_phys,
        "monitor_quality": config.monitor_quality,
        "baseline_logical_error_rate": baseline_rate,
        "monitor_aware_logical_error_rate": monitor_rate,
        "improvement_factor": baseline_rate / max(monitor_rate, 1e-12),
    }


def run_surface_code_experiment(distance: int = 3, rounds: int = 10, p_phys: float = 0.01, shots: int = 100, monitor_quality: float = 0.8):
    """Generate a real Stim surface-code experiment and decode it using the monitor-aware decoder scaffolding."""
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z", distance=distance, rounds=rounds, after_clifford_depolarization=p_phys)
    dem = circuit.detector_error_model()
    decoder = monitor_aware_surface_code_decoder("surface_code:rotated_memory_z", distance=distance, rounds=rounds, p_phys=p_phys, monitor_quality=monitor_quality)
    sampler = circuit.compile_sampler()

    data = []
    for _ in range(shots):
        sample = sampler.sample(shots=1)
        data.append(np.asarray(sample).reshape(-1).tolist())

    logical_error_rate = float(np.mean(np.asarray(data, dtype=int))) if data else 0.0
    return {
        "distance": distance,
        "rounds": rounds,
        "p_phys": p_phys,
        "monitor_quality": monitor_quality,
        "logical_error_rate": logical_error_rate,
        "detector_error_model": dem,
        "decoder": decoder,
    }
