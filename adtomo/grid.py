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
    Coordinates use ``(x, y, z)`` = ``(East, North, Down)``. Local scalar
    fields use ``(z, y, x)`` = ``(Down, North, East)`` tensor order.
    """

    def __init__(self, station_spherical, events_spherical, model, spacing):
        self.spacing = float(spacing)
        reference = model.vp
        station_spherical = torch.as_tensor(
            station_spherical, dtype=reference.dtype, device=reference.device
        ).reshape(3)
        events_spherical = torch.as_tensor(
            events_spherical, dtype=reference.dtype, device=reference.device
        ).reshape(-1, 3)
        station_ecef = spherical_to_ecef(*station_spherical)
        basis = local_basis(station_spherical[0], station_spherical[1])
        events_ecef = spherical_to_ecef(
            events_spherical[:, 0], events_spherical[:, 1], events_spherical[:, 2]
        )
        station_local = torch.zeros(3, dtype=reference.dtype, device=reference.device)
        events_local = ecef_to_local(events_ecef, station_ecef, basis)

        local_points = torch.cat([station_local[None], events_local], dim=0)
        padding = 2.0 * self.spacing
        minimum = local_points.amin(dim=0)
        maximum = local_points.amax(dim=0)
        n_west = math.ceil(float((padding - minimum[0]) / self.spacing))
        n_east = math.ceil(float((maximum[0] + padding) / self.spacing))
        n_south = math.ceil(float((padding - minimum[1]) / self.spacing))
        n_north = math.ceil(float((maximum[1] + padding) / self.spacing))
        n_up = math.ceil(float(-minimum[2] / self.spacing))
        n_down = math.ceil(float((maximum[2] + padding) / self.spacing))
        self.x = torch.arange(-n_west, n_east + 1, dtype=reference.dtype, device=reference.device) * self.spacing
        self.y = torch.arange(-n_south, n_north + 1, dtype=reference.dtype, device=reference.device) * self.spacing
        self.z = torch.arange(-n_up, n_down + 1, dtype=reference.dtype, device=reference.device) * self.spacing
        self.shape = (len(self.z), len(self.y), len(self.x))
        self.station_index = torch.tensor(
            [n_west, n_south, n_up], dtype=reference.dtype, device=reference.device
        )
        self.events_index = torch.stack(
            [
                (events_local[:, 0] - self.x[0]) / self.spacing,
                (events_local[:, 1] - self.y[0]) / self.spacing,
                (events_local[:, 2] - self.z[0]) / self.spacing,
            ],
            dim=-1,
        )

        z_local, y_local, x_local = torch.meshgrid(self.z, self.y, self.x, indexing="ij")
        local_points = torch.stack([x_local, y_local, z_local], dim=-1)
        grid_ecef = local_to_ecef(local_points, station_ecef, basis)
        grid_lon, grid_lat, grid_depth = ecef_to_spherical(grid_ecef)
        self._check_model_coverage(model, grid_lon, grid_lat, grid_depth)
        self.sample_grid = torch.stack(
            [
                self._normalize(grid_lon, model.lon),
                self._normalize(grid_lat, model.lat),
                self._normalize(grid_depth, model.depth),
            ],
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

    def sample_model(self, model_field):
        """Differentiably sample a global ``(depth, latitude, longitude)`` field."""
        return F.grid_sample(
            model_field[None, None], self.sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0]

    def sample_events(self, traveltime, event_indices=None):
        """Sample a local ``(z_local, y_local, x_local)`` field at events."""
        index = self.events_index
        if event_indices is not None:
            index = index[event_indices]
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        event_grid = torch.stack(
            [2.0 * index[:, 0] / (nx - 1) - 1.0, 2.0 * index[:, 1] / (ny - 1) - 1.0, 2.0 * index[:, 2] / (nz - 1) - 1.0],
            dim=-1,
        ).view(1, -1, 1, 1, 3)
        return F.grid_sample(traveltime[None, None], event_grid, mode="bilinear", align_corners=True)[0, 0, :, 0, 0]
