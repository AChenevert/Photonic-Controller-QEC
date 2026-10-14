from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pymatching
import stim


@dataclass(frozen=True)
class BackendSamples:
    """Backend-neutral detector, observable, and optional monitor shot data."""

    detection_events: np.ndarray
    observable_flips: np.ndarray
    monitor_observations: np.ndarray | None = None

    def __post_init__(self) -> None:
        detection_events = np.asarray(self.detection_events, dtype=int)
        observable_flips = np.asarray(self.observable_flips, dtype=int)
        if detection_events.ndim != 2 or observable_flips.ndim != 2:
            raise ValueError("detection events and observable flips must be 2D shot arrays")
        if detection_events.shape[0] != observable_flips.shape[0]:
            raise ValueError("detector and observable arrays must contain the same shots")
        if any(np.any((array != 0) & (array != 1)) for array in (detection_events, observable_flips)):
            raise ValueError("detector and observable samples must be binary")
        object.__setattr__(self, "detection_events", detection_events)
        object.__setattr__(self, "observable_flips", observable_flips)
        if self.monitor_observations is not None:
            monitor_observations = np.asarray(self.monitor_observations)
            if monitor_observations.ndim != 2 or monitor_observations.shape[0] != detection_events.shape[0]:
                raise ValueError("monitor observations must be a 2D array with one row per shot")
            if not np.all(np.isfinite(monitor_observations)):
                raise ValueError("monitor observations must be finite")
            object.__setattr__(self, "monitor_observations", monitor_observations)


class SurfaceCodeBackend(Protocol):
    """Contract for Stim, Qiskit, hardware, or another qubit simulator adapter."""

    def sample(self, shots: int, seed: int | None = None) -> BackendSamples: ...

    def decode(
        self,
        detection_events: np.ndarray,
        monitor_observations: np.ndarray | None = None,
    ) -> np.ndarray: ...


@dataclass
class StimMemoryBackend:
    distance: int = 3
    rounds: int = 10
    p_phys: float = 0.01
    code_task: str = "surface_code:rotated_memory_z"
    circuit: stim.Circuit = field(init=False, repr=False)
    detector_error_model: stim.DetectorErrorModel = field(init=False, repr=False)
    matcher: pymatching.Matching = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.distance < 2 or self.rounds < 1:
            raise ValueError("distance must be at least 2 and rounds must be positive")
        if not np.isfinite(self.p_phys) or not 0.0 <= self.p_phys < 1.0:
            raise ValueError("p_phys must be a finite probability in [0, 1)")
        self.circuit = stim.Circuit.generated(
            self.code_task,
            distance=self.distance,
            rounds=self.rounds,
            after_clifford_depolarization=self.p_phys,
        )
        self.detector_error_model = self.circuit.detector_error_model()
        self.matcher = pymatching.Matching.from_detector_error_model(self.detector_error_model)

    def sample(self, shots: int, seed: int | None = None) -> BackendSamples:
        if shots < 1:
            raise ValueError("shots must be positive")
        detector_samples, observable_samples = self.circuit.compile_detector_sampler(
            seed=seed
        ).sample(
            shots=shots,
            separate_observables=True,
        )
        return BackendSamples(detector_samples, observable_samples)

    def decode(
        self,
        detection_events: np.ndarray,
        monitor_observations: np.ndarray | None = None,
    ) -> np.ndarray:
        predictions = np.asarray(self.matcher.decode_batch(detection_events), dtype=int)
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        return predictions


def run_backend_experiment(
    backend: SurfaceCodeBackend,
    shots: int,
    seed: int | None = None,
) -> dict[str, object]:
    """Score decoded logical observables without coupling the runner to a backend."""
    if shots < 1:
        raise ValueError("shots must be positive")
    samples = backend.sample(shots=shots, seed=seed)
    predictions = np.asarray(
        backend.decode(samples.detection_events, samples.monitor_observations), dtype=int
    )
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if predictions.shape != samples.observable_flips.shape:
        raise ValueError("backend decoder output must match the observable-flip array shape")
    if np.any((predictions != 0) & (predictions != 1)):
        raise ValueError("backend decoder predictions must be binary")
    observable_failures = predictions ^ samples.observable_flips
    shot_failures = np.any(observable_failures, axis=1)
    return {
        "backend": type(backend).__name__,
        "shots": int(shots),
        "logical_error_rate": float(np.mean(shot_failures)),
        "observable_error_rates": np.mean(observable_failures, axis=0).tolist(),
        "samples": samples,
        "predictions": predictions,
    }