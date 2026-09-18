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
    """Vp/Vs parameterized only by spherical depth."""

    def __init__(self, depth, vp, vs, trainable=True):
        super().__init__()
        depth = torch.as_tensor(depth, dtype=torch.float64, device="cpu")
        vp = torch.as_tensor(vp, dtype=torch.float64, device="cpu")
        vs = torch.as_tensor(vs, dtype=torch.float64, device="cpu")
        if depth.ndim != 1 or depth.numel() < 2:
            raise ValueError("depth must be a 1-D axis with at least two nodes")
        if vp.shape != depth.shape or vs.shape != depth.shape:
            raise ValueError("vp and vs must have the same shape as depth")
        if not torch.isfinite(depth).all() or not torch.all(depth[1:] > depth[:-1]):
            raise ValueError("depth must be finite and strictly increasing")
        if not torch.isfinite(vp).all() or not torch.isfinite(vs).all() or torch.any(vp <= 0) or torch.any(vs <= 0):
            raise ValueError("Vp and Vs must be finite and positive")
        self.register_buffer("depth", depth)
        self.vp = nn.Parameter(vp.clone(), requires_grad=trainable)
        self.vs = nn.Parameter(vs.clone(), requires_grad=trainable)


def interpolate_1d(axis, values, query):
    """Piecewise-linear interpolation of ``values`` on a strictly increasing 1-D ``axis``."""
    if torch.any(query < axis[0]) or torch.any(query > axis[-1]):
        raise ValueError(
            f"query depth [{query.min().item():.3f}, {query.max().item():.3f}] km exceeds model depth "
            f"[{axis[0].item():.3f}, {axis[-1].item():.3f}] km"
        )
    upper = torch.searchsorted(axis, query.contiguous()).clamp(1, axis.numel() - 1)
    lower = upper - 1
    weight = (query - axis[lower]) / (axis[upper] - axis[lower])
    return (1.0 - weight) * values[lower] + weight * values[upper]


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
        upper = torch.tensor([nx - 1, ny - 1, nz - 1], dtype=index.dtype, device=index.device)
        if torch.any(index < 0) or torch.any(index > upper):
            raise ValueError("current event location left the fixed 3-D forward grid")
        event_grid = torch.stack(
            [2.0 * index[:, 0] / (nx - 1) - 1.0, 2.0 * index[:, 1] / (ny - 1) - 1.0, 2.0 * index[:, 2] / (nz - 1) - 1.0],
            dim=-1,
        ).view(1, -1, 1, 1, 3)
        return F.grid_sample(traveltime[None, None], event_grid, mode="bilinear", align_corners=True)[0, 0, :, 0, 0]


class ForwardGrid2D:
    """Station-centered Cartesian vertical section for a spherical 1-D Earth.

    Physical coordinates ``(x, y)`` = (horizontal section coordinate, Down),
    station at ``(0, 0)``; fields use ``field[y, x]`` order. The section is
    the azimuthal reduction of the 3-D East/North/Down frame that
    :class:`ForwardGrid` uses: an event with local coordinates ``(E, N, D)``
    sits at ``(sqrt(E^2 + N^2), D)``, and a node ``(x, 0, y)`` takes the
    spherical depth of that local point, ``d(x, y) = R - sqrt(x^2 + (R - d_s -
    y)^2)``, so spherical layers stay curved in the section. Node depths are
    fixed at construction; event positions are never cached but mapped live by
    :meth:`sample_events`, so trainable hypocenters differentiate through the
    same mapping. Because nodes at ``y = 0, x != 0`` lie above the spherical
    surface, the 1-D model needs a small negative-depth halo.

    ``padding`` (km, default ``2 * spacing``) is added left of the station,
    beyond the farthest event, and below the deepest point; like
    :class:`ForwardGrid`, nothing is added above the shallowest point.
    """

    def __init__(self, station_spherical, initial_events_spherical, model, spacing, padding=None):
        self.spacing = float(spacing)
        if not math.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        reference = model.vp
        station = torch.as_tensor(station_spherical, dtype=reference.dtype, device=reference.device).reshape(3)
        self.station_spherical = station
        self.station_ecef = spherical_to_ecef(*station)
        self.basis = local_basis(station[0], station[1])
        events_xy = self.event_xy(initial_events_spherical)

        padding = 2.0 * self.spacing if padding is None else float(padding)
        n_left = max(1, math.ceil(padding / self.spacing))
        n_right = max(1, math.ceil(float(events_xy[:, 0].max() + padding) / self.spacing))
        n_up = math.ceil(-min(0.0, float(events_xy[:, 1].min())) / self.spacing)
        n_down = max(1, math.ceil((max(0.0, float(events_xy[:, 1].max())) + padding) / self.spacing))
        self.x = torch.arange(-n_left, n_right + 1, dtype=reference.dtype, device=reference.device) * self.spacing
        self.y = torch.arange(-n_up, n_down + 1, dtype=reference.dtype, device=reference.device) * self.spacing
        self.shape = (len(self.y), len(self.x))
        # eikonal2d_op takes the source as fractional (x, y) grid indices.
        self.station_index = torch.stack([-self.x[0] / self.spacing, -self.y[0] / self.spacing])

        y_grid, x_grid = torch.meshgrid(self.y, self.x, indexing="ij")
        local_xyz = torch.stack([x_grid, torch.zeros_like(x_grid), y_grid], dim=-1)
        _, _, self.grid_depth = ecef_to_spherical(local_to_ecef(local_xyz, self.station_ecef, self.basis))
        if torch.any(self.grid_depth < model.depth[0]) or torch.any(self.grid_depth > model.depth[-1]):
            raise ValueError(
                f"2-D forward grid requires spherical depths [{self.grid_depth.min().item():.3f}, "
                f"{self.grid_depth.max().item():.3f}] km, but the 1-D model provides "
                f"[{model.depth[0].item():.3f}, {model.depth[-1].item():.3f}] km"
            )

    def event_xy(self, events_spherical):
        """Azimuthal reduction of the 3-D local frame: ``(E, N, D) -> (sqrt(E^2 + N^2), D)``."""
        events = torch.as_tensor(
            events_spherical, dtype=self.station_ecef.dtype, device=self.station_ecef.device
        ).reshape(-1, 3)
        events_ecef = spherical_to_ecef(events[:, 0], events[:, 1], events[:, 2])
        events_local = ecef_to_local(events_ecef, self.station_ecef, self.basis)
        return torch.stack([torch.hypot(events_local[:, 0], events_local[:, 1]), events_local[:, 2]], dim=-1)

    def sample_model(self, model_field, model_depth):
        """Differentiably map a depth-only field onto the section at each node's spherical depth."""
        return interpolate_1d(model_depth, model_field, self.grid_depth)

    def sample_events(self, traveltime, events_spherical):
        """Bilinearly sample a ``(y, x)`` field at live (possibly trainable) event positions."""
        xy = self.event_xy(events_spherical)
        index_x = (xy[:, 0] - self.x[0]) / self.spacing
        index_y = (xy[:, 1] - self.y[0]) / self.spacing
        nx, ny = len(self.x), len(self.y)
        if torch.any(index_x < 0) or torch.any(index_x > nx - 1) or torch.any(index_y < 0) or torch.any(index_y > ny - 1):
            raise ValueError("current event location left the fixed 2-D forward grid")
        grid = torch.stack([2.0 * index_x / (nx - 1) - 1.0, 2.0 * index_y / (ny - 1) - 1.0], dim=-1).view(1, -1, 1, 2)
        return F.grid_sample(traveltime[None, None], grid, mode="bilinear", align_corners=True)[0, 0, :, 0]
