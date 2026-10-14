import numpy as np
import pytest

from qec_monitor import MRMFrequencyResponse, MRMMonitorModel


def test_frequency_response_interpolates_complex_samples_without_extrapolating():
    response = MRMFrequencyResponse(
        frequencies_hz=np.array([100.0, 200.0, 300.0]),
        transfer=np.array([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j]),
    )

    assert response.at(150.0) == pytest.approx(2.0 + 3.0j)
    np.testing.assert_allclose(response.at(np.array([100.0, 300.0])), [1.0 + 2.0j, 5.0 + 6.0j])
    with pytest.raises(ValueError, match="outside"):
        response.at(99.0)


def test_monitor_accepts_arbitrary_pulse_envelope_and_calibrated_frequency_data():
    times = np.linspace(-2.0, 2.0, 401)
    envelope = np.exp(-times**2) * np.exp(0.1j * times)
    response = MRMFrequencyResponse(
        frequencies_hz=np.array([190.0, 195.0, 200.0]),
        transfer=np.array([0.1 - 0.05j, 0.2 + 0.0j, 0.1 + 0.05j]),
    )
    model = MRMMonitorModel(
        input_amplitude=1.0,
        frequency_response=response,
        monitor_gain=1.0,
    )
    frequency_hz = 195.0
    signal = model.drop_port_signal(
        times,
        pulse_envelope=envelope,
        carrier_frequency_hz=frequency_hz,
        amplitude_error=0.1,
        phase_error=0.2,
    )
    reference = envelope * response.at(frequency_hz)
    metrics = model.estimate_control_error(
        signal,
        times,
        reference_input=reference,
        carrier_frequency_hz=frequency_hz,
    )

    assert metrics.amplitude == pytest.approx(1.1, rel=1e-6)
    assert metrics.phase == pytest.approx(0.2, abs=1e-6)
    assert metrics.pulse_area > 0.0


def test_assumed_lorentzian_response_changes_with_carrier_frequency():
    model = MRMMonitorModel(
        ring_coupling=0.2,
        resonance_frequency_hz=100.0,
        linewidth_hz=20.0,
    )
    on_resonance = model._drop_port_transfer(100.0)
    half_linewidth_away = model._drop_port_transfer(110.0)

    assert abs(on_resonance) == pytest.approx(0.2)
    assert abs(half_linewidth_away) == pytest.approx(0.2 / np.sqrt(2.0))


def test_monitor_rejects_invalid_time_and_frequency_response_grids():
    with pytest.raises(ValueError, match="strictly increasing"):
        MRMFrequencyResponse([1.0, 1.0], [1.0 + 0j, 2.0 + 0j])
    with pytest.raises(ValueError, match="strictly increasing"):
        MRMMonitorModel().drop_port_signal(np.array([0.0, 1.0, 0.5]))