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
- `src/qec_monitor/simulator.py`: repetition-code experiment driver.

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

## Notes

This is a scaffold for a research project. It provides a minimal executable model and a clearly separated simulation architecture that can be extended with richer Stim/Qiskit circuits and serious MWPM studies.
