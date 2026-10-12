# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from adtomo.eikonal3d import Eikonal3D

# %%
if __name__ == "__main__":

    # %%
    ######################################## Settings #########################################
    ## inverted variables, each with its own shape:
    ##   1d: Vs(depth) + one Vp/Vs;  3d: Vs(lon, lat, depth) + one Vp/Vs;
    ##   3d1d: Vs(lon, lat, depth) + Vp/Vs(depth);  3d3d: both fields (lon, lat, depth)
    model = os.environ.get("MODEL", "1d")
    noise = float(os.environ.get("NOISE", 0.02))  # Gaussian pick noise (s)
    lambda_smooth = float(os.environ.get("LAMBDA", 0.0 if noise == 0 else 10.0 if model == "1d" else 100.0))
    max_iter = int(os.environ.get("MAX_ITER", 100))
    tag = f"{model}_{'noisy' if noise > 0 else 'clean'}"
    result_path = "results/layered3d_sph"
    figure_path = "figures/layered3d_sph"
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    # %%
    ######################################## Layered Model #########################################
    np.random.seed(0)
    nlon, nlat, ndepth = 16, 16, 16
    lon = np.linspace(-122.15, -121.95, nlon)  # degrees
    lat = np.linspace(37.70, 37.90, nlat)  # degrees
    depth = np.arange(ndepth) * 1.0  # km, positive down
    vpvs_ratio = 1.73
    layer_top = np.array([0.0, 4.0, 8.0, 12.0])  # km
    layer_vp = np.array([5.0, 5.8, 6.4, 7.0])  # km/s
    vp_depth = layer_vp[np.searchsorted(layer_top, depth, side="right") - 1]
    vp_true = np.broadcast_to(vp_depth[None, None, :], (nlon, nlat, ndepth)).copy()
    vs_true = vp_true / vpvs_ratio
    vpvs_ratio0 = 1.80  # homogeneous starting model: Vp 6.0 km/s, Vp/Vs 1.80
    vs0 = 6.0 / vpvs_ratio0

    # %%
    ######################################## Synthetic Data #########################################
    num_station = 16
    num_event = 50
    station_lonlat = np.random.default_rng(3).uniform(
        [lon[1], lat[1]], [lon[3], lat[-2]], (num_station, 2)
    )
    stations = pd.DataFrame(
        {
            "station_id": [f"STA{i:02d}" for i in range(num_station)],
            "longitude": station_lonlat[:, 0],
            "latitude": station_lonlat[:, 1],
            "depth_km": 0.5
            * (1.0 + np.sin(2 * np.pi * (station_lonlat[:, 0] - lon[0]) / (lon[-1] - lon[0]))),
            "dt_s": 0.0,
        }
    )
    events = pd.DataFrame(
        {
            "event_id": np.arange(num_event),
            "longitude": np.random.uniform(lon[1], lon[-2], num_event),
            "latitude": np.random.uniform(lat[1], lat[-2], num_event),
            "depth_km": np.random.uniform(3.0, depth[-2], num_event),
            "event_time": 0.0,  # travel time == arrival time
        }
    )
    events["idx_eve"] = np.arange(num_event)
    stations["idx_sta"] = np.arange(num_station)
    picks = pd.DataFrame(
        [
            {
                "event_id": event.event_id,
                "station_id": station.station_id,
                "phase_type": phase_type,
                "phase_time": 0.0,
                "idx_eve": event.idx_eve,
                "idx_sta": station.idx_sta,
            }
            for station in stations.itertuples()
            for event in events.itertuples()
            for phase_type in ("P", "S")
        ]
    )

    
    config = {"nx": 8, "ny": 8, "nz": 14, "h": 2}
    true_model = Eikonal3D(
        num_event,
        num_station,
        stations[["longitude", "latitude", "depth_km"]].values,
        stations[["dt_s"]].values,
        events[["longitude", "latitude", "depth_km"]].values,
        events[["event_time"]].values,
        vp_true,
        vs_true,
        config=config,
        lon=lon,
        lat=lat,
        depth=depth,
    )
    with torch.no_grad():
        synthetic, _ = true_model(picks)
    picks["phase_time"] = synthetic["pred"].values
    picks["phase_time"] += np.random.default_rng(1).normal(0.0, noise, len(picks))
    print(f"{num_station} stations, {num_event} events, {len(picks)} picks; {model}, noise {noise} s, lambda {lambda_smooth}")

    # %%
    ######################################## Inversion #########################################
    spherical_shape = (nlon, nlat, ndepth)
    vs_shape, ratio_shape = {
        "1d": (ndepth, ()),
        "3d": (spherical_shape, ()),
        "3d1d": (spherical_shape, ndepth),
        "3d3d": (spherical_shape, spherical_shape),
    }[model]
    sigma_t = max(noise, 1e-3)  # pick uncertainty; noise-free data get a nominal 1 ms
    eikonal3d = Eikonal3D(
        num_event,
        num_station,
        stations[["longitude", "latitude", "depth_km"]].values,
        stations[["dt_s"]].values,
        events[["longitude", "latitude", "depth_km"]].values,  # true locations, held fixed
        events[["event_time"]].values,
        None,  # vp is vpvs_ratio * vs
        np.full(vs_shape, vs0),
        vpvs_ratio=np.full(ratio_shape, vpvs_ratio0),
        sigma_t=sigma_t,
        lambda_smooth=lambda_smooth,
        config=config,
        lon=lon,
        lat=lat,
        depth=depth,
    )
    eikonal3d.event_loc.weight.requires_grad = False
    eikonal3d.event_time.weight.requires_grad = False

    parameters = [param for param in eikonal3d.parameters() if param.requires_grad]
    optimizer = optim.LBFGS(params=parameters, max_iter=max_iter, line_search_fn="strong_wolfe")
    history = {"loss": [], "misfit": []}  # misfit: mean (residual / sigma_t)^2, the data term of the loss

    def closure():
        optimizer.zero_grad()
        preds, loss = eikonal3d(picks)
        loss.backward()
        history["loss"].append(loss.item())
        history["misfit"].append(np.mean(((preds["pred"].values - picks["phase_time"].values) / sigma_t) ** 2))
        return loss

    t0 = time.time()
    optimizer.step(closure)
    print(f"Inversion: {time.time() - t0:.1f}s, {len(history['loss'])} evaluations, data misfit {history['misfit'][-1]:.3g}")

    # %%
    ######################################## Results #########################################
    vp_inv, vs_inv = [v.detach().numpy() for v in eikonal3d.velocity()]
    with torch.no_grad():
        _, vs_true_cartesian = true_model.velocity()
    vs_true_cartesian = vs_true_cartesian.detach().numpy()
    print(f"Vp/Vs: true {vpvs_ratio}, inverted {(vp_inv / vs_inv).min():.4f} to {(vp_inv / vs_inv).max():.4f}")
    print(f"Vs RMS error: {np.sqrt(np.mean((vs_inv - vs_true_cartesian) ** 2)):.3f} km/s")
    print(f"Final data misfit: {history['misfit'][-1]:.3g}")
    np.savez(
        f"{result_path}/{tag}.npz",
        lon=lon,
        lat=lat,
        depth=depth,
        vp_true=vp_true,
        vs_true=vs_true,
        vp_inv=vp_inv,
        vs_inv=vs_inv,
    )

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.semilogy(history["loss"], "k", label="Total loss")
    ax.semilogy(history["misfit"], label="Data misfit, mean (r/σ)²")
    if lambda_smooth > 0:
        ax.semilogy(np.array(history["loss"]) - np.array(history["misfit"]), label="Regularization")
    if noise > 0:
        ax.axhline(1.0, color="gray", linestyle="--", label="Noise level")
    ax.set_xlabel("Function evaluation")
    ax.set_title(f"{tag}, lambda = {lambda_smooth:g}")
    ax.legend()
    plt.savefig(f"{figure_path}/{tag}_loss.png", bbox_inches="tight", dpi=150)

    # %%
    ######################################## Plots #########################################
    ## panels: longitude-depth at middle latitude (true, inverted), map at depth, depth profile
    depth_map = 6.0  # km, inside the second layer
    ilat, idepth = nlat // 2, int(round(depth_map / (depth[1] - depth[0])))
    layouts = {
        "1d": [[("vs", "profile")]],
        "3d": [[("vs", "true"), ("vs", "inverted"), ("vs", "map"), ("vs", "profile")]],
        "3d1d": [[("vs", "true"), ("vs", "inverted"), ("vs", "map"), ("vs", "profile"), ("vpvs_ratio", "profile")]],
        "3d3d": [
            [("vs", "true"), ("vs", "inverted"), ("vs", "map"), ("vs", "profile")],
            [("vpvs_ratio", "true"), ("vpvs_ratio", "inverted"), ("vpvs_ratio", "map"), ("vpvs_ratio", "profile")],
        ],
    }
    inverted = {name: (eikonal3d.offset[name] + torch.exp(x)).detach().numpy() for name, x in eikonal3d.params.items()}
    truth = {"vs": vs_true, "vpvs_ratio": vp_true / vs_true}
    initial = {"vs": vs0, "vpvs_ratio": vpvs_ratio0}
    label = {"vs": "Vs (km/s)", "vpvs_ratio": "Vp/Vs"}
    dlon, dlat, ddepth = lon[1] - lon[0], lat[1] - lat[0], depth[1] - depth[0]
    extent_section = [lon[0] - dlon / 2, lon[-1] + dlon / 2, depth[-1] + ddepth / 2, depth[0] - ddepth / 2]
    extent_map = [lon[0] - dlon / 2, lon[-1] + dlon / 2, lat[-1] + dlat / 2, lat[0] - dlat / 2]

    layout = layouts[model]
    width = [1.3 if panel != "profile" else 0.8 for _, panel in max(layout, key=len)]
    fig, axes = plt.subplots(
        len(layout),
        len(width),
        figsize=(4 * sum(width), 4.5 * len(layout)),
        squeeze=False,
        layout="constrained",
        gridspec_kw={"width_ratios": width},
    )
    title = f"{tag}, lambda = {lambda_smooth:g}"
    if inverted["vpvs_ratio"].ndim == 0:
        title += f"; Vp/Vs {inverted['vpvs_ratio']:.4f} (true {vpvs_ratio}, initial {vpvs_ratio0})"
    fig.suptitle(title)
    for row, ax_row in zip(layout, axes):
        for (name, panel), ax in zip(row, ax_row):
            value = inverted[name]
            vmin = min(truth[name].min(), value.min(), initial[name])
            vmax = max(truth[name].max(), value.max(), initial[name])
            if panel in ("true", "inverted"):
                field = truth[name] if panel == "true" else value
                im = ax.imshow(field[:, ilat, :].T, cmap="jet_r", vmin=vmin, vmax=vmax, origin="upper", extent=extent_section)
                ax.plot(events["longitude"], events["depth_km"], "k.", markersize=3)
                ax.plot(stations["longitude"], stations["depth_km"], "k^", markersize=6)
                ax.set_xlabel("Longitude (degrees)")
                ax.set_ylabel("Depth (km)")
                ax.set_title(f"{panel.capitalize()} {label[name]}, latitude = {lat[ilat]:.3f}°")
                fig.colorbar(im, ax=ax, shrink=0.8)
            elif panel == "map":
                im = ax.imshow(value[:, :, idepth].T, cmap="jet_r", vmin=vmin, vmax=vmax, origin="upper", extent=extent_map)
                ax.plot(events["longitude"], events["latitude"], "k.", markersize=3)
                ax.plot(stations["longitude"], stations["latitude"], "k^", markersize=6)
                ax.set_xlabel("Longitude (degrees)")
                ax.set_ylabel("Latitude (degrees)")
                ax.set_title(f"Inverted {label[name]}, depth = {depth[idepth]:g} km")
                fig.colorbar(im, ax=ax, shrink=0.8)
            else:
                ax.plot(
                    np.repeat(truth[name][0, 0, :], 2),
                    np.stack([depth - ddepth / 2, depth + ddepth / 2], axis=1).ravel(),
                    "k",
                    label="True",
                )
                ax.axvline(initial[name], color="gray", linestyle="--", label="Initial")
                if value.ndim == 1:
                    ax.plot(value, depth, "r.-", label="Inverted")
                else:
                    interior = value[1:-1, 1:-1, :].reshape(-1, ndepth).T
                    ax.plot(interior.mean(axis=1), depth, "r.-", label="Inverted (mean over lon, lat)")
                    ax.fill_betweenx(
                        depth,
                        interior.min(axis=1),
                        interior.max(axis=1),
                        color="r",
                        alpha=0.2,
                        label="Range over lon, lat",
                    )
                ax.set_ylim(depth[-1] + ddepth / 2, depth[0] - ddepth / 2)
                ax.set_xlabel(label[name])
                ax.set_ylabel("Depth (km)")
                ax.set_title(f"{label[name]} profile")
                ax.legend(loc="lower left", fontsize=8)
        for ax in ax_row[len(row) :]:
            ax.axis("off")
    plt.savefig(f"{figure_path}/{tag}.png", bbox_inches="tight", dpi=150)

