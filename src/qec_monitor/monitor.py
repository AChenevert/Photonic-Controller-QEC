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


@dataclass(frozen=True)
class MRMFrequencyResponse:
    """Measured or simulated complex drop-port transfer samples versus frequency."""

    frequencies_hz: np.ndarray
    transfer: np.ndarray

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        transfer = np.asarray(self.transfer, dtype=complex)
        if frequencies.ndim != 1 or transfer.shape != frequencies.shape or frequencies.size < 2:
            raise ValueError("frequency response requires matching 1D arrays with at least two samples")
        if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(transfer)):
            raise ValueError("frequency-response samples must be finite")
        if np.any(np.diff(frequencies) <= 0.0):
            raise ValueError("frequencies_hz must be strictly increasing")
        object.__setattr__(self, "frequencies_hz", frequencies)
        object.__setattr__(self, "transfer", transfer)

    def at(self, frequency_hz: float | np.ndarray) -> complex | np.ndarray:
        frequencies = np.asarray(frequency_hz, dtype=float)
        if not np.all(np.isfinite(frequencies)):
            raise ValueError("requested frequencies must be finite")
        if np.any(frequencies < self.frequencies_hz[0]) or np.any(
            frequencies > self.frequencies_hz[-1]
        ):
            raise ValueError("requested frequency is outside the supplied response data")
        real = np.interp(frequencies, self.frequencies_hz, np.real(self.transfer))
        imag = np.interp(frequencies, self.frequencies_hz, np.imag(self.transfer))
        response = real + 1j * imag
        return complex(response) if frequencies.ndim == 0 else response


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
        resonance_frequency_hz: float = 193.5e12,
        linewidth_hz: float = 10e9,
        frequency_response: MRMFrequencyResponse | None = None,
    ) -> None:
        model_values = (
            input_amplitude,
            sigma,
            center,
            ring_coupling,
            monitor_gain,
            noise_std,
            resonance_frequency_hz,
            linewidth_hz,
        )
        if not np.all(np.isfinite(model_values)):
            raise ValueError("MRM model parameters must be finite")
        if sigma <= 0.0 or linewidth_hz <= 0.0:
            raise ValueError("sigma and linewidth_hz must be positive")
        if ring_coupling < 0.0 or monitor_gain < 0.0 or noise_std < 0.0:
            raise ValueError("ring_coupling, monitor_gain, and noise_std must be nonnegative")
        self.input_amplitude = input_amplitude
        self.sigma = sigma
        self.center = center
        self.ring_coupling = ring_coupling
        self.monitor_gain = monitor_gain
        self.noise_std = noise_std
        self.resonance_frequency_hz = resonance_frequency_hz
        self.linewidth_hz = linewidth_hz
        self.frequency_response = frequency_response

    def _validated_times(self, times: np.ndarray) -> np.ndarray:
        values = np.asarray(times, dtype=float)
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError("times must be a finite 1D array with at least two samples")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("times must be strictly increasing")
        return values

    def _drop_port_transfer(self, frequency_hz: float) -> complex:
        if not np.isfinite(frequency_hz):
            raise ValueError("carrier_frequency_hz must be finite")
        if self.frequency_response is not None:
            return complex(self.frequency_response.at(frequency_hz))
        normalized_detuning = 2.0 * (frequency_hz - self.resonance_frequency_hz) / self.linewidth_hz
        return complex(self.ring_coupling / (1.0 + 1j * normalized_detuning))

    def gaussian_input(self, times: np.ndarray) -> np.ndarray:
        times = self._validated_times(times)
        return self.input_amplitude * np.exp(-0.5 * ((times - self.center) / self.sigma) ** 2)

    def drop_port_signal(
        self,
        times: np.ndarray,
        amplitude_error: float = 0.0,
        phase_error: float = 0.0,
        detuning: float = 0.0,
        drift: float = 0.0,
        pulse_envelope: np.ndarray | None = None,
        carrier_frequency_hz: float | None = None,
        add_noise: bool = False,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        times = self._validated_times(times)
        if pulse_envelope is None:
            envelope = self.gaussian_input(times)
        else:
            envelope = np.asarray(pulse_envelope, dtype=complex)
            if envelope.shape != times.shape or not np.all(np.isfinite(envelope)):
                raise ValueError("pulse_envelope must be finite and match the time samples")
        frequency = (
            self.resonance_frequency_hz
            if carrier_frequency_hz is None
            else float(carrier_frequency_hz)
        )
        input_signal = envelope * (1.0 + amplitude_error)
        modulation = np.exp(1j * (phase_error + detuning * times))
        signal = (
            self.monitor_gain
            * input_signal
            * self._drop_port_transfer(frequency)
            * (1.0 + drift)
            * modulation
        )
        if add_noise and self.noise_std > 0.0:
            generator = rng if rng is not None else np.random.default_rng()
            noise = generator.normal(size=times.shape) + 1j * generator.normal(size=times.shape)
            signal = signal + (self.noise_std / np.sqrt(2.0)) * noise
        return signal

    def estimate_control_error(
        self,
        measured_drop: np.ndarray,
        times: np.ndarray,
        reference_input: np.ndarray | None = None,
        carrier_frequency_hz: float | None = None,
    ) -> PulseMetrics:
        times = self._validated_times(times)
        frequency = (
            self.resonance_frequency_hz
            if carrier_frequency_hz is None
            else float(carrier_frequency_hz)
        )
        if reference_input is None:
            reference_input = (
                self.gaussian_input(times)
                * self.monitor_gain
                * self._drop_port_transfer(frequency)
            )

        ref = np.asarray(reference_input, dtype=complex)
        signal = np.asarray(measured_drop, dtype=complex)
        if ref.shape != times.shape or signal.shape != times.shape:
            raise ValueError("reference and measured signals must match the time samples")
        if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(signal)):
            raise ValueError("reference and measured signals must be finite")

        overlap = np.trapezoid(signal * np.conj(ref), times)
        ref_energy = float(np.trapezoid(np.abs(ref) ** 2, times))
        amplitude_factor = abs(overlap) / max(ref_energy, 1e-12)
        phase_estimate = float(np.angle(overlap + 1e-12))
        pulse_area = float(np.trapezoid(np.abs(signal), times))
        signal_energy = float(np.trapezoid(np.abs(signal) ** 2, times))
        timing_offset = float(
            np.trapezoid(times * np.abs(signal) ** 2, times)
            / max(signal_energy, 1e-12)
        )

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
