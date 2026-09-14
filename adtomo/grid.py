"""Station-specific Cartesian forward grids for a spherical velocity model."""

import math

import torch
import torch.nn.functional as F

from .coordinate import ecef_to_local, ecef_to_spherical, local_basis, local_to_ecef, spherical_to_ecef


class ForwardGrid:
    """A fixed East/North/Down forward grid for one station.

    Global model fields use ``(depth, latitude, longitude)`` tensor order.
    Local forward and travel-time fields use ``(z_local, y_local, x_local)``
    = ``(Down, North, East)``. Fractional local locations use physical order
    ``(x_local, y_local, z_local)`` = ``(East, North, Down)``.
    """

    def __init__(self, station_lonlatdepth, event_lonlatdepth, model, spacing):
        self.spacing = float(spacing)
        ref = model.vp
        station_lonlatdepth = torch.as_tensor(station_lonlatdepth, dtype=ref.dtype, device=ref.device).reshape(3)
        event_lonlatdepth = torch.as_tensor(event_lonlatdepth, dtype=ref.dtype, device=ref.device).reshape(-1, 3)
        station_ecef = spherical_to_ecef(*station_lonlatdepth)
        basis = local_basis(station_lonlatdepth[0], station_lonlatdepth[1])
        event_ecef = spherical_to_ecef(
            event_lonlatdepth[:, 0], event_lonlatdepth[:, 1], event_lonlatdepth[:, 2]
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
        self.event_index = (event_local - low) / self.spacing

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

    def sample(self, field):
        """Differentiably sample a global ``(depth, latitude, longitude)`` field."""
        return F.grid_sample(
            field[None, None], self.sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0]

    def sample_events(self, traveltime, event_indices=None):
        """Sample a local ``(z_local, y_local, x_local)`` field at events."""
        index = self.event_index
        if event_indices is not None:
            index = index[event_indices]
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        query = torch.stack(
            [2.0 * index[:, 0] / (nx - 1) - 1.0, 2.0 * index[:, 1] / (ny - 1) - 1.0, 2.0 * index[:, 2] / (nz - 1) - 1.0],
            dim=-1,
        ).view(1, -1, 1, 1, 3)
        return F.grid_sample(traveltime[None, None], query, mode="bilinear", align_corners=True)[0, 0, :, 0, 0]
