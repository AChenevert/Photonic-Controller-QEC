# Monitor-Aware Surface-Code QEC for Photonic Controllers

This project explores how a monitor port on a photonic control path can improve surface-code QEC by weighting decoder edges using control-path reliability estimates.

## Concept

- Data qubits are protected by a surface code.
- Ancilla checks determine syndrome bits from parity measurements.
- The readout is effectively one bit of information per ancilla.
- The real control path is a photonic signal with amplitude and phase.
- A micro-ring modulator (MRM) can be monitored at the drop port to estimate the signal sent to the modulation port.
- That monitor information is converted into a gate-specific error estimate and used to update decoder edge weights.

## Files

- `src/qec_monitor/monitor.py`: MRM monitor model and signal-estimation utilities.
- `src/qec_monitor/decoder.py`: monitor-aware weighted decoding graph.
- `src/qec_monitor/simulator.py`: exact MATLAB-matched surface-code model, plus repetition-code and Stim experiment drivers.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -c "from qec_monitor import MRMMonitorModel, SimulationConfig, run_monitor_experiment; import numpy as np; m = MRMMonitorModel(); t = np.linspace(-3, 3, 200); s = m.drop_port_signal(t, amplitude_error=0.05, phase_error=0.1); print(m.estimate_control_error(s, t)); print(run_monitor_experiment(SimulationConfig(distance=3, shots=100)))"
```

## Research goals

1. Reproduce repetition-code and surface-code syndrome physics with Stim/Qiskit.
2. Model monitor-port control reliability using MRM drop-port observations.
3. Map monitor information into dynamic edge weights for a decoder.
4. Compare baseline, oracle, noisy monitor, and shuffled-monitor conditions.
5. Study distance-3, 5, and 7 rotated surface codes under drift and burst errors.

## Exact MATLAB-matched experiment

`run_exact_surface_code_experiment()` mirrors the model in the MATLAB guide. It
uses the nine-qubit distance-3 rotated code, one independent X-error layer,
perfect checks, and exact probability sums over every error and monitor word.
The monitor reports noisy channel risk, not realized faults. This is a
probability-model teaching example; it does not simulate optical hardware,
noisy syndrome extraction, or repeated rounds.

```python
from qec_monitor import run_exact_surface_code_experiment

default = run_exact_surface_code_experiment()
stronger_anomaly = run_exact_surface_code_experiment({"p_bad": 0.10})
print(default["baseline_ler"], default["monitor_ler"])
print(stronger_anomaly["baseline_ler"], stronger_anomaly["monitor_ler"])
```

The existing `run_surface_code_experiment()` remains the separate sampled Stim
circuit path. Its logical-error score uses Stim observables, not the raw
measurement-bit average; it is currently syndrome-only and does not consume
monitor data. `run_monitor_experiment()` remains the repetition-code scaffold.

## Full validation run

Run the tests and save reproducible JSON/CSV results for the MATLAB-matched
scenarios and an assumed MRR frequency sweep:

```bash
python scripts/run_validation.py
```

Use `--output-dir PATH` to choose the report directory, or `--skip-tests` to
regenerate reports only. The suite exhaustively checks all 512 X-error words
and both logical recovery choices, the MATLAB reference rates and controls,
Stim detector/observable handling, and monitor pulse/frequency inputs.

## MRR pulse and frequency inputs

`MRMMonitorModel` accepts a caller-provided sampled `pulse_envelope` and
`carrier_frequency_hz`. Until device simulations are supplied, it uses explicit
assumptions: 193.5 THz resonance, 10 GHz linewidth, and 0.12 drop-port coupling
in a single-pole Lorentzian response. These are placeholders, not hardware
measurements. Replace the assumed response with a complex transfer curve:

```python
from qec_monitor import MRMFrequencyResponse, MRMMonitorModel

response = MRMFrequencyResponse(frequencies_hz, complex_drop_port_transfer)
monitor = MRMMonitorModel(frequency_response=response)
drop_signal = monitor.drop_port_signal(
	times,
	pulse_envelope=sampled_pulse,
	carrier_frequency_hz=carrier_frequency_hz,
)
```

The backend contract (`SurfaceCodeBackend` and `BackendSamples`) normalizes
detector events, logical-observable flips, and optional per-shot monitor
observations. Stim implements it now; a Qiskit or hardware adapter can implement
the same sampling/decoding methods later. Qiskit execution is not yet wired into
the runner, and the scalar-frequency model does not yet apply a transfer curve
across a pulse's full optical spectrum.

## Notes

This is a scaffold for a research project. It provides a minimal executable model and a clearly separated simulation architecture that can be extended with richer Stim/Qiskit circuits and serious MWPM studies.
