"""The global spherical velocity model."""

import torch
import torch.nn as nn


def _axis(name, value, like=None):
    value = torch.as_tensor(value, dtype=None if like is None else like.dtype, device=None if like is None else like.device)
    if value.ndim != 1 or value.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional axis with at least two nodes")
    if not torch.isfinite(value).all() or not torch.all(value[1:] > value[:-1]):
        raise ValueError(f"{name} must be finite and strictly increasing")
    step = value[1] - value[0]
    if not torch.allclose(value[1:] - value[:-1], torch.full_like(value[1:], step), rtol=1e-6, atol=1e-10):
        raise ValueError(f"{name} must be uniformly spaced")
    return value


class VelocityModel(nn.Module):
    """Absolute Vp/Vs on a regular ``(depth, latitude, longitude)`` grid."""

    def __init__(self, lon, lat, depth, vp, vs, trainable=True):
        super().__init__()
        vp = torch.as_tensor(vp)
        vs = torch.as_tensor(vs, dtype=vp.dtype, device=vp.device)
        if vp.ndim != 3 or tuple(vp.shape) != tuple(vs.shape):
            raise ValueError("vp and vs must have matching three-dimensional (z, y, x) shapes")
        if not torch.isfinite(vp).all() or not torch.isfinite(vs).all() or torch.any(vp <= 0) or torch.any(vs <= 0):
            raise ValueError("vp and vs must be finite and positive")
        lon = _axis("lon", lon, vp)
        lat = _axis("lat", lat, vp)
        depth = _axis("depth", depth, vp)
        if tuple(vp.shape) != (len(depth), len(lat), len(lon)):
            raise ValueError("velocity shape must be (len(depth), len(lat), len(lon))")
        self.register_buffer("lon", lon)
        self.register_buffer("lat", lat)
        self.register_buffer("depth", depth)
        self.vp = nn.Parameter(vp.clone(), requires_grad=trainable)
        self.vs = nn.Parameter(vs.clone(), requires_grad=trainable)
