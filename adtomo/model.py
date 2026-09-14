"""The global spherical velocity model."""

import torch
import torch.nn as nn

class VelocityModel(nn.Module):
    """Absolute Vp/Vs on a regular ``(depth, latitude, longitude)`` grid."""

    def __init__(self, lon, lat, depth, vp, vs, trainable=True):
        super().__init__()
        vp = torch.as_tensor(vp)
        vs = torch.as_tensor(vs, dtype=vp.dtype, device=vp.device)
        self.register_buffer("lon", torch.as_tensor(lon, dtype=vp.dtype, device=vp.device))
        self.register_buffer("lat", torch.as_tensor(lat, dtype=vp.dtype, device=vp.device))
        self.register_buffer("depth", torch.as_tensor(depth, dtype=vp.dtype, device=vp.device))
        self.vp = nn.Parameter(vp.clone(), requires_grad=trainable)
        self.vs = nn.Parameter(vs.clone(), requires_grad=trainable)
