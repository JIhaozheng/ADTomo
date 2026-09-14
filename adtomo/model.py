"""The global spherical velocity model."""

import torch
import torch.nn as nn


def _axis(name, axis, like):
    axis = torch.as_tensor(axis, dtype=like.dtype, device=like.device)
    if axis.ndim != 1 or axis.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional axis with at least two nodes")
    if not torch.isfinite(axis).all() or not torch.all(axis[1:] > axis[:-1]):
        raise ValueError(f"{name} must be finite and strictly increasing")
    step = axis[1] - axis[0]
    if not torch.allclose(axis[1:] - axis[:-1], torch.full_like(axis[1:], step), rtol=1e-6, atol=1e-10):
        raise ValueError(f"{name} must be uniformly spaced")
    return axis


class VelocityModel(nn.Module):
    """Absolute Vp/Vs on a regular ``(depth, latitude, longitude)`` grid."""

    def __init__(self, lon, lat, depth, vp, vs, trainable=True):
        super().__init__()
        vp = torch.as_tensor(vp)
        vs = torch.as_tensor(vs, dtype=vp.dtype, device=vp.device)
        lon = _axis("lon", lon, vp)
        lat = _axis("lat", lat, vp)
        depth = _axis("depth", depth, vp)
        shape = (len(depth), len(lat), len(lon))
        if tuple(vp.shape) != shape or tuple(vs.shape) != shape:
            raise ValueError("vp and vs must have shape (len(depth), len(lat), len(lon))")
        if not torch.isfinite(vp).all() or not torch.isfinite(vs).all() or torch.any(vp <= 0) or torch.any(vs <= 0):
            raise ValueError("vp and vs must be finite and positive")
        self.register_buffer("lon", lon)
        self.register_buffer("lat", lat)
        self.register_buffer("depth", depth)
        self.vp = nn.Parameter(vp.clone(), requires_grad=trainable)
        self.vs = nn.Parameter(vs.clone(), requires_grad=trainable)
