from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends import StimMemoryBackend, run_backend_experiment
from .decoder import match_and_decode
from .surface_code import rotated_surface_code_distance_3


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


def _surface_code_joint_probabilities(
    errors: np.ndarray,
    joint_index: np.ndarray,
    qubit_error_rates: np.ndarray,
) -> np.ndarray:
    weights = np.prod(
        np.where(errors == 1, qubit_error_rates, 1.0 - qubit_error_rates), axis=1
    )
    joint = np.bincount(joint_index, weights=weights, minlength=32).reshape(2, 16)
    assert abs(float(np.sum(joint)) - 1.0) < 1e-12
    return joint


def _evaluate_exact_monitor(
    parameters: dict[str, float],
    errors: np.ndarray,
    monitors: np.ndarray,
    joint_index: np.ndarray,
    prior_joint: np.ndarray,
    baseline_decisions: np.ndarray,
) -> dict[str, float]:
    risk_fraction = parameters["bad_fraction"]
    false_positive = parameters["false_positive"]
    false_negative = parameters["false_negative"]
    alarm_probability = (
        risk_fraction * (1.0 - false_negative)
        + (1.0 - risk_fraction) * false_positive
    )
    risk_given_alarm = (
        risk_fraction * (1.0 - false_negative) / alarm_probability
        if alarm_probability > 0.0
        else risk_fraction
    )
    risk_given_quiet = (
        risk_fraction * false_negative / (1.0 - alarm_probability)
        if alarm_probability < 1.0
        else risk_fraction
    )
    error_if_alarm = parameters["p_good"] + (
        parameters["p_bad"] - parameters["p_good"]
    ) * risk_given_alarm
    error_if_quiet = parameters["p_good"] + (
        parameters["p_bad"] - parameters["p_good"]
    ) * risk_given_quiet

    monitor_weights = np.prod(
        np.where(monitors == 1, alarm_probability, 1.0 - alarm_probability), axis=1
    )
    conditioned_ler = 0.0
    shuffled_ler = 0.0
    baseline_recomputed = 0.0
    reconstructed_joint = np.zeros((2, 16), dtype=float)
    syndrome_ids = np.arange(16)

    for monitor, monitor_weight in zip(monitors, monitor_weights):
        if monitor_weight == 0.0:
            continue
        error_rates = np.where(monitor == 1, error_if_alarm, error_if_quiet)
        joint = _surface_code_joint_probabilities(errors, joint_index, error_rates)
        decisions = np.argmax(joint, axis=0)
        wrong_rows = 1 - decisions
        conditioned_ler += monitor_weight * float(np.sum(joint[wrong_rows, syndrome_ids]))
        shuffled_ler += monitor_weight * float(
            np.sum(prior_joint[wrong_rows, syndrome_ids])
        )
        baseline_recomputed += monitor_weight * float(
            np.sum(joint[1 - baseline_decisions, syndrome_ids])
        )
        reconstructed_joint += monitor_weight * joint

    assert np.max(np.abs(reconstructed_joint - prior_joint)) < 1e-12
    baseline_ler = float(np.sum(prior_joint[1 - baseline_decisions, syndrome_ids]))
    assert abs(baseline_recomputed - baseline_ler) < 1e-12
    return {
        "monitor_ler": conditioned_ler,
        "shuffled_monitor_ler": shuffled_ler,
        "alarm_probability": alarm_probability,
        "p_alarm": error_if_alarm,
        "p_quiet": error_if_quiet,
    }


def run_exact_surface_code_experiment(
    parameters: dict[str, float] | None = None,
) -> dict[str, object]:
    """Exactly enumerate the MATLAB guide's d=3, X-only monitor-assisted code."""
    values = {
        "bad_fraction": 0.05,
        "p_good": 0.001,
        "p_bad": 0.05,
        "false_positive": 0.02,
        "false_negative": 0.20,
    }
    if parameters is not None:
        if not isinstance(parameters, dict):
            raise ValueError("parameters must be a dictionary of probabilities")
        unknown = set(parameters) - set(values)
        if unknown:
            raise ValueError(f"unknown parameter: {sorted(unknown)[0]}")
        values.update(parameters)

    for name, value in values.items():
        scalar = np.asarray(value)
        if scalar.ndim != 0 or scalar.dtype.kind not in "iuf" or not np.isfinite(scalar):
            raise ValueError(f"{name} must be a finite probability in [0, 1]")
        values[name] = float(scalar)
        if not 0.0 <= values[name] <= 1.0:
            raise ValueError(f"{name} must be a finite probability in [0, 1]")
    if values["p_good"] > values["p_bad"] or values["p_bad"] >= 0.5:
        raise ValueError("require 0 <= p_good <= p_bad < 0.5")

    code = rotated_surface_code_distance_3()
    errors = code.error_words()
    monitors = errors.copy()
    syndrome_bits = code.syndrome(errors)
    syndrome_ids = syndrome_bits @ (2 ** np.arange(4))
    logical_classes = code.logical_class(errors)
    joint_index = syndrome_ids + 16 * logical_classes
    assert np.all(np.bincount(joint_index, minlength=32) == 16)

    average_error_rate = (
        (1.0 - values["bad_fraction"]) * values["p_good"]
        + values["bad_fraction"] * values["p_bad"]
    )
    prior_joint = _surface_code_joint_probabilities(
        errors, joint_index, np.full(9, average_error_rate)
    )
    baseline_decisions = np.argmax(prior_joint, axis=0)
    syndrome_ids = np.arange(16)
    baseline_ler = float(
        np.sum(prior_joint[1 - baseline_decisions, syndrome_ids])
    )

    sensor = _evaluate_exact_monitor(
        values, errors, monitors, joint_index, prior_joint, baseline_decisions
    )
    perfect_values = {**values, "false_positive": 0.0, "false_negative": 0.0}
    perfect = _evaluate_exact_monitor(
        perfect_values, errors, monitors, joint_index, prior_joint, baseline_decisions
    )
    uninformative_values = {**values, "false_positive": 0.5, "false_negative": 0.5}
    uninformative = _evaluate_exact_monitor(
        uninformative_values, errors, monitors, joint_index, prior_joint, baseline_decisions
    )

    tolerance = 1e-12
    assert sensor["monitor_ler"] <= baseline_ler + tolerance
    assert perfect["monitor_ler"] <= sensor["monitor_ler"] + tolerance
    assert abs(uninformative["monitor_ler"] - baseline_ler) < tolerance
    assert sensor["shuffled_monitor_ler"] >= baseline_ler - tolerance

    return {
        "scope": "d=3 rotated surface code; X-only data noise; perfect checks; one layer",
        "method": "Exact probability sums; optimal logical-class decoding; no postselection",
        "parameters": values,
        "average_physical_X_probability": average_error_rate,
        "baseline_ler": baseline_ler,
        "monitor_ler": sensor["monitor_ler"],
        "shuffled_monitor_ler": sensor["shuffled_monitor_ler"],
        "perfect_risk_monitor_ler": perfect["monitor_ler"],
        "uninformative_monitor_ler": uninformative["monitor_ler"],
        "fault_probability_if_alarm": sensor["p_alarm"],
        "fault_probability_if_no_alarm": sensor["p_quiet"],
        "relative_reduction": 1.0 - sensor["monitor_ler"] / baseline_ler
        if baseline_ler > 0.0
        else 0.0,
        "all_builtin_checks_passed": True,
    }


def run_surface_code_experiment(
    distance: int = 3,
    rounds: int = 10,
    p_phys: float = 0.01,
    shots: int = 100,
    monitor_quality: float = 0.8,
    seed: int | None = None,
):
    """Run Stim memory shots and score decoded logical-observable flips.

    The current Stim path is syndrome-only. ``monitor_quality`` is retained for
    compatibility; monitor-conditioned matching will be supplied by a decoder
    adapter that consumes per-shot monitor observations.
    """
    backend = StimMemoryBackend(distance=distance, rounds=rounds, p_phys=p_phys)
    experiment = run_backend_experiment(backend, shots=shots, seed=seed)
    return {
        "distance": distance,
        "rounds": rounds,
        "p_phys": p_phys,
        "monitor_quality": monitor_quality,
        "logical_error_rate": experiment["logical_error_rate"],
        "observable_error_rates": experiment["observable_error_rates"],
        "detector_error_model": backend.detector_error_model,
        "decoder": backend.matcher,
        "backend_samples": experiment["samples"],
        "monitor_used": False,
    }
