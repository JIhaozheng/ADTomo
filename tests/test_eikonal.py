from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from adtomo import solve_eikonal2d, solve_eikonal3d


def assert_value_error(message, function):
    try:
        function()
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_small_2d_and_3d_solves():
    t2 = solve_eikonal2d(torch.full((5, 7), 2.0, dtype=torch.float64), (1.2, 2.1), 1.0)
    t3 = solve_eikonal3d(torch.full((4, 5, 6), 2.0, dtype=torch.float64), (1.2, 2.1, 1.3), 1.0)
    assert t2.shape == (5, 7)
    assert t3.shape == (4, 5, 6)
    assert torch.isfinite(t2).all() and torch.isfinite(t3).all()


def test_solver_rejects_bad_inputs():
    assert_value_error("float64", lambda: solve_eikonal3d(torch.ones((3, 3, 3)), (1, 1, 1), 1.0))
    assert_value_error(
        "positive", lambda: solve_eikonal2d(torch.zeros((3, 3), dtype=torch.float64), (1, 1), 1.0)
    )
    assert_value_error(
        "inside", lambda: solve_eikonal3d(torch.ones((3, 3, 3), dtype=torch.float64), (2, 1, 1), 1.0)
    )


def main():
    test_small_2d_and_3d_solves()
    test_solver_rejects_bad_inputs()
    print(f"{Path(__file__).name}: passed")


if __name__ == "__main__":
    main()
