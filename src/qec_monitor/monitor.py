from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt


@dataclass
class PulseMetrics:
    amplitude: float
    phase: float
    pulse_area: float
    timing_offset: float
    detuning: float
    channel_drift: float


def gaussian_pulse_qutip(times: np.ndarray, center: float = 0.0, sigma: float = 1.0) -> qt.Qobj:
    """Create a Gaussian envelope using a Qobj for visualization and analysis."""
    envelope = np.exp(-0.5 * ((times - center) / sigma) ** 2)
    return qt.Qobj(np.asarray([envelope], dtype=complex))


class MRMMonitorModel:
    """Simple photonic monitor model for an MRM-driven control channel.

    The input line carries a constant-amplitude Gaussian pulse. The micro-ring
    modulator (MRM) modulates that signal and the monitor reads the drop-port
    field. By comparing the measured drop-port signal to the expected reference,
    we estimate amplitude, phase, timing, detuning, and drift for each control
    channel.
    """

    def __init__(
        self,
        input_amplitude: float = 1.0,
        sigma: float = 0.8,
        center: float = 0.0,
        ring_coupling: float = 0.12,
        monitor_gain: float = 1.0,
        noise_std: float = 0.01,
    ) -> None:
        self.input_amplitude = input_amplitude
        self.sigma = sigma
        self.center = center
        self.ring_coupling = ring_coupling
        self.monitor_gain = monitor_gain
        self.noise_std = noise_std

    def gaussian_input(self, times: np.ndarray) -> np.ndarray:
        return self.input_amplitude * np.exp(-0.5 * ((times - self.center) / self.sigma) ** 2)

    def drop_port_signal(
        self,
        times: np.ndarray,
        amplitude_error: float = 0.0,
        phase_error: float = 0.0,
        detuning: float = 0.0,
        drift: float = 0.0,
    ) -> np.ndarray:
        input_signal = self.gaussian_input(times) * (1.0 + amplitude_error)
        modulation = np.exp(1j * (phase_error + detuning * times))
        ring_response = self.ring_coupling * (1.0 + drift)
        return self.monitor_gain * input_signal * ring_response * modulation

    def estimate_control_error(
        self,
        measured_drop: np.ndarray,
        times: np.ndarray,
        reference_input: np.ndarray | None = None,
    ) -> PulseMetrics:
        if reference_input is None:
            reference_input = self.gaussian_input(times)

        ref = np.asarray(reference_input, dtype=complex)
        signal = np.asarray(measured_drop, dtype=complex)

        overlap = np.trapz(signal * np.conj(ref), times)
        ref_energy = np.trapz(ref * np.conj(ref), times)
        amplitude_factor = abs(overlap) / max(ref_energy, 1e-12)
        phase_estimate = float(np.angle(overlap + 1e-12))
        pulse_area = float(np.trapz(np.abs(signal), times))
        timing_offset = float(np.trapz(times * np.abs(signal) ** 2, times) / max(np.trapz(np.abs(signal) ** 2, times), 1e-12))

        return PulseMetrics(
            amplitude=float(amplitude_factor),
            phase=phase_estimate,
            pulse_area=pulse_area,
            timing_offset=timing_offset,
            detuning=0.0,
            channel_drift=float(amplitude_factor - 1.0),
        )

    def accuracy_score(self, metrics: PulseMetrics) -> float:
        """Map the inferred control error into a monitor-quality score in [0, 1]."""
        amplitude_error = abs(metrics.channel_drift)
        phase_error = abs(metrics.phase)
        return float(np.clip(1.0 - (amplitude_error + phase_error / np.pi) / 2.0, 0.0, 1.0))
