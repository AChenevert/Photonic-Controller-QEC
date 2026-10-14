from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceCodeModel:
    """Binary stabilizer data needed by exact and sampled surface-code backends."""

    name: str
    distance: int
    x_checks: np.ndarray
    z_checks: np.ndarray
    logical_x: np.ndarray
    logical_z: np.ndarray

    def __post_init__(self) -> None:
        x_checks = np.asarray(self.x_checks, dtype=int)
        z_checks = np.asarray(self.z_checks, dtype=int)
        logical_x = np.asarray(self.logical_x, dtype=int)
        logical_z = np.asarray(self.logical_z, dtype=int)
        if x_checks.ndim != 2 or z_checks.ndim != 2 or x_checks.shape[1] != z_checks.shape[1]:
            raise ValueError("X and Z check matrices must be 2D with the same data-qubit width")
        if logical_x.shape != (x_checks.shape[1],) or logical_z.shape != logical_x.shape:
            raise ValueError("logical operators must have one entry per data qubit")
        if any(np.any((array != 0) & (array != 1)) for array in (x_checks, z_checks, logical_x, logical_z)):
            raise ValueError("surface-code operators must be binary")
        object.__setattr__(self, "x_checks", x_checks)
        object.__setattr__(self, "z_checks", z_checks)
        object.__setattr__(self, "logical_x", logical_x)
        object.__setattr__(self, "logical_z", logical_z)

    @property
    def n_data(self) -> int:
        return int(self.logical_x.size)

    def error_words(self) -> np.ndarray:
        values = np.arange(1 << self.n_data, dtype=np.uint64)
        shifts = np.arange(self.n_data, dtype=np.uint64)
        return ((values[:, None] >> shifts) & 1).astype(int)

    def syndrome(self, errors: np.ndarray) -> np.ndarray:
        words = self._validate_words(errors)
        return (words @ self.z_checks.T) % 2

    def logical_class(self, errors: np.ndarray) -> np.ndarray:
        words = self._validate_words(errors)
        return (words @ self.logical_z) % 2

    def minimum_weight_recovery(self, syndrome: np.ndarray, logical_class: int) -> np.ndarray:
        syndrome_bits = np.asarray(syndrome, dtype=int)
        if syndrome_bits.shape != (self.z_checks.shape[0],) or np.any(
            (syndrome_bits != 0) & (syndrome_bits != 1)
        ):
            raise ValueError("syndrome must be a binary vector with one bit per Z check")
        if logical_class not in (0, 1):
            raise ValueError("logical_class must be 0 or 1")
        errors = self.error_words()
        candidates = errors[
            np.all(self.syndrome(errors) == syndrome_bits, axis=1)
            & (self.logical_class(errors) == logical_class)
        ]
        if candidates.size == 0:
            raise ValueError("no recovery exists for the requested syndrome and logical class")
        return candidates[int(np.argmin(np.sum(candidates, axis=1)))].copy()

    def validate(self) -> None:
        if np.any((self.x_checks @ self.z_checks.T) % 2):
            raise ValueError("X and Z checks must commute")
        if np.any((self.x_checks @ self.logical_z) % 2):
            raise ValueError("logical Z must commute with X checks")
        if np.any((self.z_checks @ self.logical_x) % 2):
            raise ValueError("logical X must commute with Z checks")
        if int((self.logical_x @ self.logical_z) % 2) != 1:
            raise ValueError("logical X and Z must anticommute")
        if self.x_checks.shape[0] != 4 or self.z_checks.shape[0] != 4:
            raise ValueError("the exact MATLAB-matched code requires four checks of each type")

        check_words = np.arange(1 << 4, dtype=np.uint8)[:, None]
        check_generators = ((check_words >> np.arange(4, dtype=np.uint8)) & 1).astype(int)
        if np.unique((check_generators @ self.x_checks) % 2, axis=0).shape[0] != 16:
            raise ValueError("X checks must be independent")
        if np.unique((check_generators @ self.z_checks) % 2, axis=0).shape[0] != 16:
            raise ValueError("Z checks must be independent")

        errors = self.error_words()
        for checks, logical in ((self.z_checks, self.logical_z), (self.x_checks, self.logical_x)):
            trivial_syndrome = np.all((errors @ checks.T) % 2 == 0, axis=1)
            nontrivial_logical = (errors @ logical) % 2 == 1
            if np.min(np.sum(errors[trivial_syndrome & nontrivial_logical], axis=1)) != self.distance:
                raise ValueError("code distance does not match the declared distance")

        syndrome_ids = self.syndrome(errors) @ (2 ** np.arange(self.z_checks.shape[0]))
        logical_classes = self.logical_class(errors)
        buckets = syndrome_ids + (1 << self.z_checks.shape[0]) * logical_classes
        if not np.all(np.bincount(buckets, minlength=32) == 16):
            raise ValueError("each syndrome/logical class must contain 16 equivalent error words")

    def _validate_words(self, errors: np.ndarray) -> np.ndarray:
        words = np.asarray(errors, dtype=int)
        if words.ndim not in (1, 2) or words.shape[-1] != self.n_data:
            raise ValueError("error words must have one bit per data qubit")
        if np.any((words != 0) & (words != 1)):
            raise ValueError("error words must be binary")
        return words


def rotated_surface_code_distance_3() -> SurfaceCodeModel:
    """Return the exact nine-data-qubit layout defined in the MATLAB guide."""
    x_checks = np.zeros((4, 9), dtype=int)
    z_checks = np.zeros((4, 9), dtype=int)
    x_supports = ((1, 2, 4, 5), (5, 6, 8, 9), (2, 3), (7, 8))
    z_supports = ((2, 3, 5, 6), (4, 5, 7, 8), (1, 4), (6, 9))
    for row, support in enumerate(x_supports):
        x_checks[row, np.asarray(support) - 1] = 1
    for row, support in enumerate(z_supports):
        z_checks[row, np.asarray(support) - 1] = 1
    logical_x = np.zeros(9, dtype=int)
    logical_x[[0, 3, 6]] = 1
    logical_z = np.zeros(9, dtype=int)
    logical_z[[0, 1, 2]] = 1
    model = SurfaceCodeModel(
        name="rotated_surface_code_d3",
        distance=3,
        x_checks=x_checks,
        z_checks=z_checks,
        logical_x=logical_x,
        logical_z=logical_z,
    )
    model.validate()
    return model