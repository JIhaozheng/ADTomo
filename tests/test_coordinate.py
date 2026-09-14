from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from adtomo.coordinate import ecef_to_local, ecef_to_spherical, local_basis, local_to_ecef, spherical_to_ecef


def test_spherical_ecef_roundtrip_and_basis():
    points = torch.tensor([[-120.0, 35.0, 0.0], [-119.7, 34.8, 12.0]], dtype=torch.float64)
    xyz = spherical_to_ecef(points[:, 0], points[:, 1], points[:, 2])
    lon, lat, depth = ecef_to_spherical(xyz)
    assert torch.allclose(torch.stack([lon, lat, depth], dim=-1), points, atol=1e-10)
    basis = local_basis(torch.tensor(-120.0, dtype=torch.float64), torch.tensor(35.0, dtype=torch.float64))
    assert torch.allclose(basis @ basis.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
    local = ecef_to_local(xyz, xyz[0], basis)
    assert torch.allclose(local_to_ecef(local, xyz[0], basis), xyz, atol=1e-10)


def test_coordinate_path_has_gradients():
    points = torch.tensor([[-120.0, 35.0, 8.0]], dtype=torch.float64, requires_grad=True)
    xyz = spherical_to_ecef(points[:, 0], points[:, 1], points[:, 2])
    xyz.square().sum().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
