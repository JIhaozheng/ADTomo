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

    def __init__(self, station_lonlatdepth, event_lonlatdepth, model, spacing, padding=None):
        if model.vp.device.type != "cpu" or model.vp.dtype != torch.float64:
            raise ValueError("ForwardGrid requires a CPU torch.float64 VelocityModel")
        if not isinstance(spacing, (float, int)) or spacing <= 0:
            raise ValueError("spacing must be one positive isotropic scalar in km")
        self.spacing = float(spacing)
        self.padding = 2.0 * self.spacing if padding is None else float(padding)
        if self.padding < self.spacing:
            raise ValueError("padding must leave at least one cell around the station")

        ref = model.vp
        self.model_shape = tuple(model.vp.shape)
        self.station_lonlatdepth = torch.as_tensor(
            station_lonlatdepth, dtype=ref.dtype, device=ref.device
        ).reshape(3)
        self.event_lonlatdepth = torch.as_tensor(
            event_lonlatdepth, dtype=ref.dtype, device=ref.device
        ).reshape(-1, 3)
        if (
            len(self.event_lonlatdepth) == 0
            or not torch.isfinite(self.station_lonlatdepth).all()
            or not torch.isfinite(self.event_lonlatdepth).all()
        ):
            raise ValueError("station and at least one finite event are required")

        self.station_ecef = spherical_to_ecef(*self.station_lonlatdepth)
        self.basis = local_basis(self.station_lonlatdepth[0], self.station_lonlatdepth[1])
        event_ecef = spherical_to_ecef(
            self.event_lonlatdepth[:, 0], self.event_lonlatdepth[:, 1], self.event_lonlatdepth[:, 2]
        )
        self.station_local = torch.zeros(3, dtype=ref.dtype, device=ref.device)
        self.event_local = ecef_to_local(event_ecef, self.station_ecef, self.basis)

        points = torch.cat([self.station_local[None], self.event_local], dim=0)
        low = points.amin(dim=0) - self.padding
        high = points.amax(dim=0) + self.padding
        nxyz = tuple(max(2, math.ceil(float((high[i] - low[i]) / self.spacing)) + 1) for i in range(3))
        self.origin = low
        self.x = low[0] + torch.arange(nxyz[0], dtype=ref.dtype, device=ref.device) * self.spacing
        self.y = low[1] + torch.arange(nxyz[1], dtype=ref.dtype, device=ref.device) * self.spacing
        self.z = low[2] + torch.arange(nxyz[2], dtype=ref.dtype, device=ref.device) * self.spacing
        self.shape = (len(self.z), len(self.y), len(self.x))
        self.station_index = self._index(self.station_local)
        self.event_index = self._index(self.event_local)
        self._check_index(self.station_index, source=True)
        self._check_index(self.event_index)

        z_local, y_local, x_local = torch.meshgrid(self.z, self.y, self.x, indexing="ij")
        xyz_local = torch.stack([x_local, y_local, z_local], dim=-1)
        ecef = local_to_ecef(xyz_local, self.station_ecef, self.basis)
        lon, lat, depth = ecef_to_spherical(ecef)
        self._check_model_coverage(model, lon, lat, depth)
        self.model_grid = torch.stack(
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

    def _index(self, local):
        return (local - self.origin) / self.spacing

    def _check_index(self, index, source=False):
        upper = torch.tensor([len(self.x) - 1, len(self.y) - 1, len(self.z) - 1], dtype=index.dtype, device=index.device)
        if source:
            inside = torch.all(index >= 0) and torch.all(index < upper)
        else:
            inside = torch.all(index >= 0) and torch.all(index <= upper)
        if not inside:
            raise ValueError("station or event lies outside the forward grid")

    def sample(self, field):
        """Differentiably sample a global ``(depth, latitude, longitude)`` field."""
        if tuple(field.shape) != self.model_shape:
            raise ValueError(f"global field shape {tuple(field.shape)} != model shape {self.model_shape}")
        if field.device.type != "cpu" or field.dtype != torch.float64:
            raise ValueError("forward-grid sampling requires CPU torch.float64 fields")
        return F.grid_sample(field[None, None], self.model_grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]

    def _live_event_index(self, event_lonlatdepth):
        event_lonlatdepth = torch.as_tensor(
            event_lonlatdepth,
            dtype=self.station_lonlatdepth.dtype,
            device=self.station_lonlatdepth.device,
        ).reshape(-1, 3)
        ecef = spherical_to_ecef(
            event_lonlatdepth[:, 0], event_lonlatdepth[:, 1], event_lonlatdepth[:, 2]
        )
        index = self._index(ecef_to_local(ecef, self.station_ecef, self.basis))
        self._check_index(index)
        return index

    def sample_events(self, traveltime, event_indices=None, events=None):
        """Sample a local ``(z_local, y_local, x_local)`` field at events."""
        if tuple(traveltime.shape) != self.shape:
            raise ValueError(f"traveltime shape {tuple(traveltime.shape)} != grid shape {self.shape}")
        if events is not None and event_indices is not None:
            raise ValueError("pass event_indices or events, not both")
        index = self._live_event_index(events) if events is not None else self.event_index
        if event_indices is not None:
            index = index[torch.as_tensor(event_indices, dtype=torch.long, device=index.device)]
        self._check_index(index)
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        query = torch.stack(
            [2.0 * index[:, 0] / (nx - 1) - 1.0, 2.0 * index[:, 1] / (ny - 1) - 1.0, 2.0 * index[:, 2] / (nz - 1) - 1.0],
            dim=-1,
        ).view(1, -1, 1, 1, 3)
        return F.grid_sample(traveltime[None, None], query, mode="bilinear", align_corners=True)[0, 0, :, 0, 0]
