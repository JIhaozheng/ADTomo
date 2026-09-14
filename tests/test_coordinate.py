from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from adtomo.coordinate import ecef_to_local, ecef_to_spherical, local_basis, local_to_ecef, spherical_to_ecef


points_lonlatdepth = torch.tensor([[-120.0, 35.0, 0.0], [-119.7, 34.8, 12.0]], dtype=torch.float64)
xyz_ecef = spherical_to_ecef(
    points_lonlatdepth[:, 0], points_lonlatdepth[:, 1], points_lonlatdepth[:, 2]
)
lon, lat, depth = ecef_to_spherical(xyz_ecef)
assert torch.allclose(torch.stack([lon, lat, depth], dim=-1), points_lonlatdepth, atol=1e-10)

basis = local_basis(torch.tensor(-120.0, dtype=torch.float64), torch.tensor(35.0, dtype=torch.float64))
assert torch.allclose(basis @ basis.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
xyz_local = ecef_to_local(xyz_ecef, xyz_ecef[0], basis)
assert torch.allclose(local_to_ecef(xyz_local, xyz_ecef[0], basis), xyz_ecef, atol=1e-10)

gradient_point_lonlatdepth = torch.tensor([[-120.0, 35.0, 8.0]], dtype=torch.float64, requires_grad=True)
gradient_xyz_ecef = spherical_to_ecef(
    gradient_point_lonlatdepth[:, 0], gradient_point_lonlatdepth[:, 1], gradient_point_lonlatdepth[:, 2]
)
(gradient_xyz_ecef[..., 0].sum() + 0.3 * gradient_xyz_ecef[..., 1].sum()).backward()
assert gradient_point_lonlatdepth.grad is not None
assert torch.isfinite(gradient_point_lonlatdepth.grad).all()
assert torch.all(gradient_point_lonlatdepth.grad.abs() > 0)

print(f"{Path(__file__).name}: passed")
