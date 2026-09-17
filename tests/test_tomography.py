"""Tomography objective and distributed-data scaling checks."""

import torch

from adtomo import ForwardGrid, Tomography, VelocityModel, predict_travel_times, smoothness


def make_model(trainable=True):
    lon = torch.arange(-120.6, -119.39, 0.1, dtype=torch.float64)
    lat = torch.arange(34.4, 35.61, 0.1, dtype=torch.float64)
    depth = torch.arange(-15.0, 45.1, 5.0, dtype=torch.float64)
    vp = torch.full((len(depth), len(lat), len(lon)), 6.0, dtype=torch.float64)
    return VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=trainable)


def make_groups(model):
    groups = []
    for station, event in (([-120.0, 35.0, 0.0], [[-119.9, 35.1, 8.0]]), ([-119.7, 35.2, 0.0], [[-119.8, 35.0, 12.0]])):
        grid = ForwardGrid(torch.tensor(station), torch.tensor(event), model, spacing=5.0)
        with torch.no_grad():
            observed_p = predict_travel_times(model, grid, "P") + 0.05
            observed_s = predict_travel_times(model, grid, "S") - 0.04
        groups.append((grid, [("P", torch.tensor([0]), observed_p), ("S", torch.tensor([0]), observed_s)]))
    return groups


def perturb(model):
    with torch.no_grad():
        model.vp[3, 4, 5] += 0.1
        model.vs[4, 5, 4] -= 0.05


def test_serial_tomography_objective_matches_data_and_regularization():
    model = make_model()
    groups = make_groups(model)
    tomography = Tomography(model, lambda_vp=0.5, lambda_vs=0.25, alpha_vp=0.125, alpha_vs=0.0625)
    perturb(model)

    loss = tomography(groups)
    residuals = [predict_travel_times(model, grid, phase, indices) - observed for grid, phase_groups in groups for phase, indices, observed in phase_groups]
    data_sum = torch.cat(residuals).square().sum()
    dvp, dvs = model.vp - tomography.vp0, model.vs - tomography.vs0
    regularization = 0.5 * smoothness(dvp, model.lon, model.lat, model.depth) + 0.25 * smoothness(dvs, model.lon, model.lat, model.depth) + 0.125 * dvp.square().mean() + 0.0625 * dvs.square().mean()
    assert torch.allclose(loss, data_sum / 4 + regularization)
    assert torch.allclose(tomography.data_sum, data_sum)
    assert torch.allclose(tomography.data_loss, data_sum / 4)
    assert torch.allclose(tomography.regularization_loss, regularization)


def test_ddp_data_scale_matches_serial_gradients():
    reference_model = make_model()
    groups = make_groups(reference_model)
    reference = Tomography(reference_model, alpha_vp=0.25, alpha_vs=0.125)
    perturb(reference_model)
    reference(groups).backward()

    gradients = []
    for local_groups in (groups[:1], groups[1:]):
        model = make_model()
        tomography = Tomography(model, alpha_vp=0.25, alpha_vs=0.125)
        perturb(model)
        tomography(local_groups, data_scale=2 / 4).backward()
        gradients.append((model.vp.grad, model.vs.grad))

    assert torch.allclose((gradients[0][0] + gradients[1][0]) / 2, reference_model.vp.grad, atol=1e-11)
    assert torch.allclose((gradients[0][1] + gradients[1][1]) / 2, reference_model.vs.grad, atol=1e-11)
