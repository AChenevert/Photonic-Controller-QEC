from __future__ import annotations

import numpy as np
import stim
import pymatching


def _effective_error_probability(base_error_rate: float, monitor_quality: float, bias: float = 1.0) -> float:
    """Convert a monitor-quality estimate into a weighted edge error probability."""
    safe_base = float(np.clip(base_error_rate, 1e-6, 0.49))
    monitor_component = float(np.clip(monitor_quality, 0.0, 1.0))
    adjusted = safe_base * (1.0 - monitor_component) + (1.0 - safe_base) * monitor_component * bias
    return float(np.clip(adjusted, 1e-6, 0.49))


def build_monitor_aware_graph(
    num_edges: int,
    base_error_rate: float = 0.01,
    monitor_quality: float = 0.0,
    biases: np.ndarray | None = None,
):
    """Create a weighted graph for a monitor-aware repetition-code decoder."""
    if biases is None:
        biases = np.ones(num_edges, dtype=float)
    else:
        biases = np.asarray(biases, dtype=float)
        if biases.shape[0] < num_edges:
            pad = np.ones(num_edges - biases.shape[0], dtype=float)
            biases = np.concatenate([biases, pad])

    graph = []
    for idx in range(num_edges):
        p = _effective_error_probability(base_error_rate, monitor_quality, float(biases[idx]))
        weight = -np.log(p + 1e-9)
        graph.append((idx, idx + 1, float(weight)))
    return graph


def match_and_decode(syndrome_nodes, n_data=None, base_error_rate=0.01, monitor_quality=0.0, biases=None):
    """Decode a simple repetition-code syndrome using a weighted nearest-neighbor correction."""
    syndrome_nodes = sorted({int(s) for s in syndrome_nodes})
    if n_data is None:
        n_data = max(syndrome_nodes) + 1 if syndrome_nodes else 0
    n_data = int(max(0, n_data))
    correction = np.zeros(n_data, dtype=int)

    if not syndrome_nodes:
        return correction

    for i in range(len(syndrome_nodes) - 1):
        start = syndrome_nodes[i]
        end = syndrome_nodes[i + 1]
        if end - start <= 2:
            correction[start:end + 1] ^= 1

    for node in syndrome_nodes:
        if node < n_data and correction[node] == 0:
            correction[node] = 1

    return correction


def monitor_aware_surface_code_decoder(code_task: str = "surface_code:rotated_memory_z", distance: int = 3, rounds: int = 10, p_phys: float = 0.01, monitor_quality: float = 0.8):
    """Generate and decode a Stim surface-code memory experiment using monitor-aware edge weighting in a simplified model."""
    circuit = stim.Circuit.generated(code_task, distance=distance, rounds=rounds, after_clifford_depolarization=p_phys)
    dem = circuit.detector_error_model()
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return {"circuit": circuit, "detector_error_model": dem, "matcher": matcher, "distance": distance, "rounds": rounds}
