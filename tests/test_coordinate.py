import torch

from adtomo.coordinate import ecef_to_local, ecef_to_spherical, local_basis, local_to_ecef, spherical_to_ecef


station_lonlatdepth = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
event_lonlatdepth = torch.tensor(
    [[-120.18, 34.90, 8.0], [-119.82, 35.08, 12.0], [-120.06, 35.22, 16.0]], dtype=torch.float64
)
points_lonlatdepth = torch.cat([station_lonlatdepth[None], event_lonlatdepth], dim=0)
points_ecef = spherical_to_ecef(points_lonlatdepth[:, 0], points_lonlatdepth[:, 1], points_lonlatdepth[:, 2])
lon, lat, depth = ecef_to_spherical(points_ecef)
assert torch.allclose(torch.stack([lon, lat, depth], dim=-1), points_lonlatdepth, atol=1e-10)

basis = local_basis(station_lonlatdepth[0], station_lonlatdepth[1])
assert torch.allclose(basis @ basis.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
event_local = ecef_to_local(points_ecef[1:], points_ecef[0], basis)
event_ecef_roundtrip = local_to_ecef(event_local, points_ecef[0], basis)
assert torch.allclose(event_ecef_roundtrip, points_ecef[1:], atol=1e-10)
assert torch.allclose(ecef_to_local(event_ecef_roundtrip, points_ecef[0], basis), event_local, atol=1e-10)

print("test_coordinate.py: passed")
