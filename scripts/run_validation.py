from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qec_monitor import MRMMonitorModel, run_exact_surface_code_experiment


def _surface_code_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for name, parameters in (
        ("default", {}),
        ("stronger_anomaly", {"p_bad": 0.10}),
        ("perfect_risk_monitor", {"false_positive": 0.0, "false_negative": 0.0}),
        ("uninformative_monitor", {"false_positive": 0.5, "false_negative": 0.5}),
        ("no_noise", {"p_good": 0.0, "p_bad": 0.0}),
        ("equal_risk", {"p_good": 0.01, "p_bad": 0.01}),
    ):
        cases.append({"name": name, **run_exact_surface_code_experiment(parameters)})

    for p_bad in (0.02, 0.05, 0.10):
        for false_positive in (0.0, 0.02, 0.10, 0.50):
            parameters = {"p_bad": p_bad, "false_positive": false_positive}
            cases.append(
                {
                    "name": f"scan_pbad_{p_bad:.2f}_fp_{false_positive:.2f}",
                    **run_exact_surface_code_experiment(parameters),
                }
            )
    return cases


def _mrr_frequency_sweep() -> dict[str, object]:
    model = MRMMonitorModel()
    times = np.linspace(-3.0, 3.0, 401)
    pulse_envelope = np.exp(-0.5 * (times / 0.6) ** 2)
    frequencies = model.resonance_frequency_hz + np.array(
        [-model.linewidth_hz, -0.5 * model.linewidth_hz, 0.0, 0.5 * model.linewidth_hz, model.linewidth_hz]
    )
    samples = []
    for frequency_hz in frequencies:
        reference_signal = model.drop_port_signal(
            times,
            pulse_envelope=pulse_envelope,
            carrier_frequency_hz=float(frequency_hz),
        )
        signal = model.drop_port_signal(
            times,
            pulse_envelope=pulse_envelope,
            carrier_frequency_hz=float(frequency_hz),
            amplitude_error=0.05,
            phase_error=0.02,
        )
        metrics = model.estimate_control_error(
            signal,
            times,
            reference_input=reference_signal,
            carrier_frequency_hz=float(frequency_hz),
        )
        samples.append(
            {
                "carrier_frequency_hz": float(frequency_hz),
                "drop_port_peak": float(np.max(np.abs(signal))),
                "estimated_amplitude_ratio": metrics.amplitude,
                "estimated_phase_rad": metrics.phase,
            }
        )
    return {
        "response_model": "assumed single-pole Lorentzian; replace with MRMFrequencyResponse data",
        "assumptions": {
            "resonance_frequency_hz": model.resonance_frequency_hz,
            "linewidth_hz": model.linewidth_hz,
            "ring_coupling": model.ring_coupling,
            "pulse": "Gaussian envelope, sigma=0.6 in model time units",
        },
        "samples": samples,
    }


def _write_reports(output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _surface_code_cases()
    frequency_sweep = _mrr_frequency_sweep()
    json_path = output_dir / "qec_validation.json"
    csv_path = output_dir / "surface_code_scan.csv"
    frequency_path = output_dir / "mrr_frequency_sweep.csv"

    json_path.write_text(
        json.dumps(
            {
                "matlab_matched_surface_code": cases,
                "mrr_frequency_sweep": frequency_sweep,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    case_fields = (
        "name",
        "average_physical_X_probability",
        "baseline_ler",
        "monitor_ler",
        "shuffled_monitor_ler",
        "perfect_risk_monitor_ler",
        "uninformative_monitor_ler",
        "relative_reduction",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=case_fields)
        writer.writeheader()
        writer.writerows({field: case[field] for field in case_fields} for case in cases)

    with frequency_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=frequency_sweep["samples"][0].keys())
        writer.writeheader()
        writer.writerows(frequency_sweep["samples"])
    return json_path, csv_path, frequency_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run QEC tests and save MATLAB-matched model reports.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-tests", action="store_true", help="Only regenerate the JSON/CSV reports.")
    arguments = parser.parse_args()

    if not arguments.skip_tests:
        environment = os.environ.copy()
        source_path = str(ROOT / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (source_path, environment.get("PYTHONPATH", "")) if value
        )
        test_run = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        if test_run.returncode:
            return test_run.returncode
    paths = _write_reports(arguments.output_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())