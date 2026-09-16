"""Global spherical velocity model and station-centered forward grids."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


R_EARTH_KM = 6371.0


class VelocityModel(nn.Module):
    """Absolute Vp/Vs on a regular ``(depth, latitude, longitude)`` grid."""

    def __init__(self, lon, lat, depth, vp, vs, trainable=True):
        super().__init__()
        vp = torch.as_tensor(vp, dtype=torch.float64, device="cpu")
        vs = torch.as_tensor(vs, dtype=torch.float64, device="cpu")
        self.register_buffer("lon", torch.as_tensor(lon, dtype=torch.float64, device="cpu"))
        self.register_buffer("lat", torch.as_tensor(lat, dtype=torch.float64, device="cpu"))
        self.register_buffer("depth", torch.as_tensor(depth, dtype=torch.float64, device="cpu"))
        self.vp = nn.Parameter(vp.clone(), requires_grad=trainable)
        self.vs = nn.Parameter(vs.clone(), requires_grad=trainable)


def spherical_to_ecef(lon, lat, depth):
    """Degrees east/north and km depth positive down to ECEF km."""
    lon = torch.deg2rad(lon)
    lat = torch.deg2rad(lat)
    radius = R_EARTH_KM - depth
    return torch.stack(
        [
            radius * torch.cos(lat) * torch.cos(lon),
            radius * torch.cos(lat) * torch.sin(lon),
            radius * torch.sin(lat),
        ],
        dim=-1,
    )


def ecef_to_spherical(xyz):
    """ECEF km to longitude, latitude, depth in degrees/degrees/km."""
    radius = torch.linalg.vector_norm(xyz, dim=-1)
    lon = torch.rad2deg(torch.atan2(xyz[..., 1], xyz[..., 0]))
    lat = torch.rad2deg(torch.atan2(xyz[..., 2], torch.hypot(xyz[..., 0], xyz[..., 1])))
    return lon, lat, R_EARTH_KM - radius


def local_basis(lon, lat):
    """Rows of the East, North, Down basis at longitude/latitude."""
    lon = torch.deg2rad(lon)
    lat = torch.deg2rad(lat)
    zero = torch.zeros_like(lon)
    east = torch.stack([-torch.sin(lon), torch.cos(lon), zero], dim=-1)
    north = torch.stack(
        [-torch.sin(lat) * torch.cos(lon), -torch.sin(lat) * torch.sin(lon), torch.cos(lat)], dim=-1
    )
    down = torch.stack(
        [-torch.cos(lat) * torch.cos(lon), -torch.cos(lat) * torch.sin(lon), -torch.sin(lat)], dim=-1
    )
    return torch.stack([east, north, down], dim=-2)


def ecef_to_local(xyz, origin, basis):
    """ECEF km to a local East/North/Down frame."""
    return (xyz - origin) @ basis.transpose(-1, -2)


def local_to_ecef(xyz, origin, basis):
    """Local East/North/Down km to ECEF."""
    return xyz @ basis + origin


class ForwardGrid:
    """A fixed East/North/Down forward grid for one station.

    Global model fields use ``(depth, latitude, longitude)`` tensor order.
    Local forward and travel-time fields use ``(z_local, y_local, x_local)``
    = ``(Down, North, East)``. Fractional local locations use physical order
    ``(x_local, y_local, z_local)`` = ``(East, North, Down)``.
    """

    def __init__(self, station_spherical, events_spherical, model, spacing):
        self.spacing = float(spacing)
        ref = model.vp
        station_spherical = torch.as_tensor(station_spherical, dtype=ref.dtype, device=ref.device).reshape(3)
        events_spherical = torch.as_tensor(events_spherical, dtype=ref.dtype, device=ref.device).reshape(-1, 3)
        station_ecef = spherical_to_ecef(*station_spherical)
        basis = local_basis(station_spherical[0], station_spherical[1])
        event_ecef = spherical_to_ecef(
            events_spherical[:, 0], events_spherical[:, 1], events_spherical[:, 2]
        )
        station_local = torch.zeros(3, dtype=ref.dtype, device=ref.device)
        event_local = ecef_to_local(event_ecef, station_ecef, basis)

        points = torch.cat([station_local[None], event_local], dim=0)
        low = points.amin(dim=0) - 2.0 * self.spacing
        high = points.amax(dim=0) + 2.0 * self.spacing
        nxyz = tuple(max(2, math.ceil(float((high[i] - low[i]) / self.spacing)) + 1) for i in range(3))
        self.x = low[0] + torch.arange(nxyz[0], dtype=ref.dtype, device=ref.device) * self.spacing
        self.y = low[1] + torch.arange(nxyz[1], dtype=ref.dtype, device=ref.device) * self.spacing
        self.z = low[2] + torch.arange(nxyz[2], dtype=ref.dtype, device=ref.device) * self.spacing
        self.shape = (len(self.z), len(self.y), len(self.x))
        self.station_index = (station_local - low) / self.spacing
        self.events_index = (event_local - low) / self.spacing

        z_local, y_local, x_local = torch.meshgrid(self.z, self.y, self.x, indexing="ij")
        xyz_local = torch.stack([x_local, y_local, z_local], dim=-1)
        ecef = local_to_ecef(xyz_local, station_ecef, basis)
        lon, lat, depth = ecef_to_spherical(ecef)
        self._check_model_coverage(model, lon, lat, depth)
        self.sample_grid = torch.stack(
            [self._normalize(lon, model.lon), self._normalize(lat, model.lat), self._normalize(depth, model.depth)],
            dim=-1,
        ).unsqueeze(0)

    @staticmethod
    def _normalize(value, axis):
        return 2.0 * (value - axis[0]) / (axis[-1] - axis[0]) - 1.0

    def _check_model_coverage(self, model, lon, lat, depth):
        values = (("longitude", lon, model.lon), ("latitude", lat, model.lat), ("depth", depth, model.depth))
        for name, value, axis in values:
            if torch.any(value < axis[0] - 1e-8) or torch.any(value > axis[-1] + 1e-8):
                raise ValueError(
                    f"forward grid needs {name} [{value.min().item():.4f}, {value.max().item():.4f}] "
                    f"but model provides [{axis[0].item():.4f}, {axis[-1].item():.4f}]"
                )

    def sample_model(self, field):
        """Differentiably sample a global ``(depth, latitude, longitude)`` field."""
        return F.grid_sample(
            field[None, None], self.sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0]

    def sample_events(self, traveltime, event_indices=None):
        """Sample a local ``(z_local, y_local, x_local)`` field at events."""
        index = self.events_index
        if event_indices is not None:
            index = index[event_indices]
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        query = torch.stack(
            [2.0 * index[:, 0] / (nx - 1) - 1.0, 2.0 * index[:, 1] / (ny - 1) - 1.0, 2.0 * index[:, 2] / (nz - 1) - 1.0],
            dim=-1,
        ).view(1, -1, 1, 1, 3)
        return F.grid_sample(traveltime[None, None], query, mode="bilinear", align_corners=True)[0, 0, :, 0, 0]
