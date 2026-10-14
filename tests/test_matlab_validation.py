import numpy as np

from qec_monitor import rotated_surface_code_distance_3


def test_surface_code_matches_matlab_stabilizer_and_distance_invariants():
    code = rotated_surface_code_distance_3()
    errors = code.error_words()

    assert errors.shape == (512, 9)
    assert np.all(np.bincount(code.logical_class(errors), minlength=2) == 256)
    assert not np.any((code.x_checks @ code.z_checks.T) % 2)


def test_all_error_words_and_both_logical_recoveries_match_matlab_validation():
    code = rotated_surface_code_distance_3()

    for error in code.error_words():
        syndrome = code.syndrome(error)
        actual_class = int(code.logical_class(error))
        for chosen_class in (0, 1):
            recovery = code.minimum_weight_recovery(syndrome, chosen_class)
            residual = error ^ recovery

            assert not np.any(code.syndrome(residual))
            assert int(code.logical_class(residual)) == (actual_class ^ chosen_class)
            assert (not np.any(code.logical_class(residual))) == (chosen_class == actual_class)


def test_every_single_qubit_x_fault_has_a_logical_correcting_recovery():
    code = rotated_surface_code_distance_3()

    for qubit in range(code.n_data):
        error = np.zeros(code.n_data, dtype=int)
        error[qubit] = 1
        recovery = code.minimum_weight_recovery(code.syndrome(error), int(code.logical_class(error)))
        assert not np.any(code.syndrome(error ^ recovery))
        assert not np.any(code.logical_class(error ^ recovery))