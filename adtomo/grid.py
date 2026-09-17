"""Global spherical velocity model and station-centered forward grids."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


R_EARTH = 6371.0 # km


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


class VelocityModel1D(nn.Module):
    """Absolute Vp/Vs on a regular, laterally invariant depth-only grid."""

    def __init__(self, depth, vp, vs, trainable=True):
        super().__init__()
        vp = torch.as_tensor(vp, dtype=torch.float64, device="cpu")
        vs = torch.as_tensor(vs, dtype=torch.float64, device="cpu")
        self.register_buffer("depth", torch.as_tensor(depth, dtype=torch.float64, device="cpu"))
        self.vp = nn.Parameter(vp.clone(), requires_grad=trainable)
        self.vs = nn.Parameter(vs.clone(), requires_grad=trainable)


def spherical_to_ecef(lon, lat, depth):
    """Degrees east/north and km depth positive down to ECEF km."""
    lon = torch.deg2rad(lon)
    lat = torch.deg2rad(lat)
    radius = R_EARTH - depth
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
    return lon, lat, R_EARTH - radius


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
        self.station_ecef = spherical_to_ecef(*station_spherical)
        self.basis = local_basis(station_spherical[0], station_spherical[1])
        events_ecef = spherical_to_ecef(
            events_spherical[:, 0], events_spherical[:, 1], events_spherical[:, 2]
        )
        station_local = torch.zeros(3, dtype=reference.dtype, device=reference.device)
        events_local = ecef_to_local(events_ecef, self.station_ecef, self.basis)

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
        self.events_index = self._local_index(events_local)

        z_local, y_local, x_local = torch.meshgrid(self.z, self.y, self.x, indexing="ij")
        local_points = torch.stack([x_local, y_local, z_local], dim=-1)
        grid_ecef = local_to_ecef(local_points, self.station_ecef, self.basis)
        grid_lon, grid_lat, grid_depth = ecef_to_spherical(grid_ecef)
        # self._check_model_coverage(model, grid_lon, grid_lat, grid_depth)
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

    def _local_index(self, events_local):
        """Convert local East/North/Down km coordinates to fractional grid indices."""
        return torch.stack(
            [
                (events_local[:, 0] - self.x[0]) / self.spacing,
                (events_local[:, 1] - self.y[0]) / self.spacing,
                (events_local[:, 2] - self.z[0]) / self.spacing,
            ],
            dim=-1,
        )

    def index_from_spherical(self, events_spherical):
        """Fractional grid indices for (possibly updated) event spherical coordinates."""
        events_spherical = torch.as_tensor(
            events_spherical, dtype=self.station_ecef.dtype, device=self.station_ecef.device
        ).reshape(-1, 3)
        events_ecef = spherical_to_ecef(events_spherical[:, 0], events_spherical[:, 1], events_spherical[:, 2])
        events_local = ecef_to_local(events_ecef, self.station_ecef, self.basis)
        return self._local_index(events_local)

    # def _check_model_coverage(self, model, lon, lat, depth):
    #     values = (("longitude", lon, model.lon), ("latitude", lat, model.lat), ("depth", depth, model.depth))
    #     for name, value, axis in values:
    #         if torch.any(value < axis[0] - 1e-8) or torch.any(value > axis[-1] + 1e-8):
    #             raise ValueError(
    #                 f"forward grid needs {name} [{value.min().item():.4f}, {value.max().item():.4f}] "
    #                 f"but model provides [{axis[0].item():.4f}, {axis[-1].item():.4f}]"
    #             )

    def sample_model(self, model_field):
        """Differentiably sample a global ``(depth, latitude, longitude)`` field."""
        return F.grid_sample(
            model_field[None, None], self.sample_grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0]

    def sample_events(self, traveltime, event_indices=None, index=None):
        """Sample a local ``(z_local, y_local, x_local)`` field at events.

        Pass ``index`` (from :meth:`index_from_spherical`) to sample at updated
        event locations instead of the grid's cached, build-time positions.
        """
        if index is None:
            index = self.events_index
            if event_indices is not None:
                index = index[event_indices]
        nx, ny, nz = len(self.x), len(self.y), len(self.z)
        event_grid = torch.stack(
            [2.0 * index[:, 0] / (nx - 1) - 1.0, 2.0 * index[:, 1] / (ny - 1) - 1.0, 2.0 * index[:, 2] / (nz - 1) - 1.0],
            dim=-1,
        ).view(1, -1, 1, 1, 3)
        return F.grid_sample(traveltime[None, None], event_grid, mode="bilinear", align_corners=True)[0, 0, :, 0, 0]


class RadialForwardGrid:
    """A station-centered ``(depth, range)`` forward grid for a 1-D model.

    A depth-only Vp/Vs(depth) model has no horizontal variation, so a ray
    between a station and any event stays in the vertical plane containing
    both, and travel time is a function of ``(depth, horizontal range)``
    alone. This lets one 2-D eikonal solve stand in for the 3-D solve that
    :class:`ForwardGrid` needs for a full 3-D model.

    ``padding`` (km, default ``2 * spacing``) is the margin added around the
    build-time event positions; widen it when events will be relocated.
    """

    def __init__(self, station_spherical, events_spherical, model, spacing, padding=None):
        self.spacing = float(spacing)
        reference = model.vp
        station_spherical = torch.as_tensor(
            station_spherical, dtype=reference.dtype, device=reference.device
        ).reshape(3)
        events_spherical = torch.as_tensor(
            events_spherical, dtype=reference.dtype, device=reference.device
        ).reshape(-1, 3)
        self.station_ecef = spherical_to_ecef(*station_spherical)
        self.basis = local_basis(station_spherical[0], station_spherical[1])
        self._model_depth = model.depth

        station_depth = station_spherical[2]
        events_range, events_depth = self._range_depth(events_spherical)

        padding = 2.0 * self.spacing if padding is None else float(padding)
        depth_min = torch.minimum(station_depth, events_depth.min()) - padding
        depth_max = torch.maximum(station_depth, events_depth.max()) + padding
        n_up = math.ceil(float((station_depth - depth_min) / self.spacing))
        n_down = math.ceil(float((depth_max - station_depth) / self.spacing))
        n_range = math.ceil(float((events_range.max() + padding) / self.spacing))

        offsets = torch.arange(-n_up, n_down + 1, dtype=reference.dtype, device=reference.device)
        self.depth = station_depth + offsets * self.spacing
        self.r = torch.arange(0, n_range + 1, dtype=reference.dtype, device=reference.device) * self.spacing
        self.shape = (len(self.depth), len(self.r))
        # (x, y) = (range index, depth index), matching eikonal2d_op's (x, y) source convention.
        self.station_index = torch.tensor([0.0, n_up], dtype=reference.dtype, device=reference.device)
        self.events_index = self._local_index(events_range, events_depth)

    def _range_depth(self, events_spherical):
        events_ecef = spherical_to_ecef(events_spherical[:, 0], events_spherical[:, 1], events_spherical[:, 2])
        events_local = ecef_to_local(events_ecef, self.station_ecef, self.basis)
        events_range = torch.hypot(events_local[:, 0], events_local[:, 1])
        return events_range, events_spherical[:, 2]

    def _local_index(self, events_range, events_depth):
        return torch.stack([events_range / self.spacing, (events_depth - self.depth[0]) / self.spacing], dim=-1)

    def index_from_spherical(self, events_spherical):
        """Fractional grid indices for (possibly updated) event spherical coordinates."""
        events_spherical = torch.as_tensor(
            events_spherical, dtype=self.station_ecef.dtype, device=self.station_ecef.device
        ).reshape(-1, 3)
        events_range, events_depth = self._range_depth(events_spherical)
        return self._local_index(events_range, events_depth)

    def sample_model(self, model_field_1d):
        """Differentiably sample a global depth-only field onto (depth, range)."""
        normalized_depth = ForwardGrid._normalize(self.depth, self._model_depth)
        grid = torch.stack([torch.zeros_like(normalized_depth), normalized_depth], dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            model_field_1d[None, None, :, None], grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0, 0, :, 0]
        return sampled[:, None].expand(-1, len(self.r))

    def sample_events(self, traveltime, event_indices=None, index=None):
        """Sample a local ``(depth, range)`` field at events.

        Pass ``index`` (from :meth:`index_from_spherical`) to sample at updated
        event locations instead of the grid's cached, build-time positions.
        """
        if index is None:
            index = self.events_index
            if event_indices is not None:
                index = index[event_indices]
        nr, nz = len(self.r), len(self.depth)
        event_grid = torch.stack(
            [2.0 * index[:, 0] / (nr - 1) - 1.0, 2.0 * index[:, 1] / (nz - 1) - 1.0], dim=-1
        ).view(1, -1, 1, 2)
        return F.grid_sample(traveltime[None, None], event_grid, mode="bilinear", align_corners=True)[0, 0, :, 0]
