"""Compare CPU float64 event interpolation implementations.

Run from this directory after building the extensions:
    python benchmark_interpolation.py
"""

import statistics
import time

import torch
import torch.nn.functional as F


torch.set_num_threads(1)
torch.manual_seed(0)
shape = (33, 35, 37)
base_field = torch.randn(shape, dtype=torch.float64)


def sample_grid(traveltime, indices):
    nx, ny, nz = traveltime.shape
    query = torch.stack(
        [
            2.0 * indices[:, 2] / (nz - 1) - 1.0,
            2.0 * indices[:, 1] / (ny - 1) - 1.0,
            2.0 * indices[:, 0] / (nx - 1) - 1.0,
        ],
        dim=-1,
    ).view(1, -1, 1, 1, 3)
    return F.grid_sample(traveltime[None, None], query, mode="bilinear", align_corners=True)[0, 0, :, 0, 0]


def sample_trilinear(traveltime, indices):
    lower = indices.floor().to(torch.long)
    fraction = indices - lower
    upper = lower + 1
    x0, y0, z0 = lower.unbind(dim=1)
    x1, y1, z1 = upper.unbind(dim=1)
    wx, wy, wz = fraction.unbind(dim=1)
    return (
        traveltime[x0, y0, z0] * (1 - wx) * (1 - wy) * (1 - wz)
        + traveltime[x0, y0, z1] * (1 - wx) * (1 - wy) * wz
        + traveltime[x0, y1, z0] * (1 - wx) * wy * (1 - wz)
        + traveltime[x0, y1, z1] * (1 - wx) * wy * wz
        + traveltime[x1, y0, z0] * wx * (1 - wy) * (1 - wz)
        + traveltime[x1, y0, z1] * wx * (1 - wy) * wz
        + traveltime[x1, y1, z0] * wx * wy * (1 - wz)
        + traveltime[x1, y1, z1] * wx * wy * wz
    )


def measure(sample, indices, repeats=10):
    forward_times = []
    backward_times = []
    for _ in range(repeats):
        field = base_field.clone().requires_grad_()
        start = time.perf_counter()
        values = sample(field, indices)
        forward_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        values.square().mean().backward()
        backward_times.append(time.perf_counter() - start)
    return statistics.median(forward_times), statistics.median(backward_times)


print("events  max|value diff|  max|gradient diff|  grid fwd/bwd (ms)  manual fwd/bwd (ms)")
for count in (1, 10, 100, 1000, 10000):
    indices = 0.1 + torch.rand((count, 3), dtype=torch.float64) * (torch.tensor(shape, dtype=torch.float64) - 1.2)
    grid_field = base_field.clone().requires_grad_()
    manual_field = base_field.clone().requires_grad_()
    grid_values = sample_grid(grid_field, indices)
    manual_values = sample_trilinear(manual_field, indices)
    grid_values.square().mean().backward()
    manual_values.square().mean().backward()
    value_error = (grid_values - manual_values).abs().max().item()
    gradient_error = (grid_field.grad - manual_field.grad).abs().max().item()
    assert value_error < 1e-12
    assert gradient_error < 1e-12
    grid_forward, grid_backward = measure(sample_grid, indices)
    manual_forward, manual_backward = measure(sample_trilinear, indices)
    print(
        f"{count:6d}  {value_error:14.3e}  {gradient_error:17.3e}  "
        f"{grid_forward * 1e3:7.3f}/{grid_backward * 1e3:7.3f}  "
        f"{manual_forward * 1e3:7.3f}/{manual_backward * 1e3:7.3f}"
    )

print("benchmark_interpolation.py: passed")
