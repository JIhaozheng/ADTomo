import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# Backend switch (same Eikonal3D API):
# from adtomo.eikonal_lu import Eikonal3D
from adtomo.eikonal_ordered import Eikonal3D
# from adtomo.eikonal_adj_0f import Eikonal3D
# from adtomo.eikonal_adj_dir import Eikonal3D

def log_timestamp():
    return f"{time.strftime('%m-%d %H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}"


## inversion demo
if __name__ == "__main__":
    ## how to run: torchrun --nproc_per_node=2 --master_port=??? 04.inversion.py

    ############## Define Parameters###########
    inversion_mode = "3d"  # "1d": only update velocity vs depth (z); "3d": full 3D update
    lambda_dvp = 0.015
    lambda_dvs = 0.003
    lambda_sp_ratio = 0
    beta_dvp = 0
    beta_dvs = 0
    beta_sp_ratio = 0
    lambda_local = 0
    n_epochs = 100
    lbfgs_max_iter = 40  # max closure evals per LBFGS step (line search)

    ######### Load event, station and picks#########
    data_path = "checkerboard"
    event_path = f"{data_path}/events.csv"
    station_path = f"{data_path}/stations.csv"
    picks_path = f"{data_path}/picks.csv"
    meta_path = f"{data_path}/checkerboard_initial.npz"
    eikonal_config_path = f"{data_path}/config.json"

    result_path = f"checkerboard/inversion"
    figure_path = f"{result_path}/figures"
    log_path = f"{result_path}/inversion.log"
    inv_vel_path = f"{result_path}/inv_model"
    final_model_path = f"{result_path}/inv_model.npz"

    with open(f"{eikonal_config_path}", "r") as f:
        eikonal_config = json.load(f)
    nx, ny, nz, h = (
        eikonal_config["nx"],
        eikonal_config["ny"],
        eikonal_config["nz"],
        eikonal_config["h"],
    )

########################## DDP setup ########################
    init_process_group(backend="gloo")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    master_process = ddp_rank == 0

    if master_process:
        path_bytes = log_path.encode("utf-8")
        path_len = torch.tensor([len(path_bytes)], dtype=torch.long, device="cpu")
    else:
        path_len = torch.tensor([0], dtype=torch.long, device="cpu")
    dist.broadcast(path_len, src=0)
    path_len = path_len.item()
    if path_len > 0:
        if master_process:
            buf = torch.ByteTensor(list(path_bytes))
        else:
            buf = torch.ByteTensor([0] * path_len)
        dist.broadcast(buf, src=0)
        if not master_process:
            log_path = bytes(buf.tolist()).decode("utf-8")

    if master_process:
        for p in (inv_vel_path, result_path, figure_path):
            os.makedirs(p, exist_ok=True)
        open(log_path, "w").close()
        with open(log_path, "a") as log_file:
            log_file.write(f"event_path: {event_path}\n")
            log_file.write(f"station_path: {station_path}\n")
            log_file.write(f"picks_path: {picks_path}\n")
            log_file.write(f"initial velocity model path: {meta_path}\n")
            log_file.write(f"eikonal_config_path: {eikonal_config_path}\n")
            log_file.write(f"result_path: {result_path}\n")
            log_file.write(f"figure_path: {figure_path}\n")
            log_file.write(f"log_path: {log_path}\n")
            log_file.write(
                f"DDP rank {ddp_rank}, local rank {ddp_local_rank}, world size {ddp_world_size}\n"
            )

########################## Inversion ########################
    events = pd.read_csv(event_path, dtype={"event_id": str})
    stations = pd.read_csv(station_path, dtype={"station_id": str})
    events["event_time"] = pd.to_datetime(events["event_time"])
    events = events[
        (events["x_km"] >= 0)
        & (events["x_km"] <= (nx - 2) * h)
        & (events["y_km"] >= 0)
        & (events["y_km"] <= (ny - 2) * h)
        & (events["z_km"] >= 0)
        & (events["z_km"] <= nz * h)
    ]
    stations = stations[
        (stations["x_km"] >= 0)
        & (stations["x_km"] <= (nx - 2) * h)
        & (stations["y_km"] >= 0)
        & (stations["y_km"] <= (ny - 2) * h)
        & (stations["z_km"] >= 0)
        & (stations["z_km"] <= nz * h)
    ]

    picks = pd.read_csv(picks_path, dtype={"event_id": str, "station_id": str})
    picks = picks.merge(events[["event_id", "event_time"]], on="event_id")

    num_event = len(events)
    num_station = len(stations)
    assert num_station >= ddp_world_size, (
        f"Number of stations ({num_station}) must be >= world size ({ddp_world_size})"
    )

    picks["phase_time_origin"] = picks["phase_time"].copy()
    picks["phase_time"] = (
        pd.to_datetime(picks["phase_time"]) - pd.to_datetime(picks["event_time"])
    ).dt.total_seconds()

    picks.drop(columns=["event_time"], inplace=True)
    events["event_time_origin"] = events["event_time"].copy()
    events["event_time"] = np.zeros(num_event)
    events["idx_eve"] = np.arange(num_event)
    stations["idx_sta"] = np.arange(num_station)
    picks = picks.merge(events[["event_id", "idx_eve"]], on="event_id")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    xgrid = torch.arange(0, nx, dtype=torch.float64) * h
    ygrid = torch.arange(0, ny, dtype=torch.float64) * h
    zgrid = torch.arange(0, nz, dtype=torch.float64) * h
    eikonal_config.update({"xgrid": xgrid, "ygrid": ygrid, "zgrid": zgrid})

    meta = np.load(meta_path)
    vp = torch.from_numpy(meta["vp"])
    vs = torch.from_numpy(meta["vs"])

    eikonal3d = Eikonal3D(
        num_event,
        num_station,
        stations[["x_km", "y_km", "z_km"]].values,
        stations[["dt_s"]].values,
        events[["x_km", "y_km", "z_km"]].values,
        events[["event_time"]].values,
        vp,
        vs,
        inversion=inversion_mode,
        lambda_dvp=lambda_dvp,
        lambda_dvs=lambda_dvs,
        lambda_sp_ratio=lambda_sp_ratio,
        beta_dvp=beta_dvp,
        beta_dvs=beta_dvs,
        beta_sp_ratio=beta_sp_ratio,
        lambda_local=lambda_local,
        config=eikonal_config,
    )

    raw_picks = picks.copy()

    # DDP: each rank only runs eikonal forward for its assigned stations
    picks = picks[picks["idx_sta"] % ddp_world_size == ddp_local_rank]
    print(
        f"Rank {ddp_rank} (local {ddp_local_rank}): {len(picks)} / {len(raw_picks)} picks",
        flush=True,
    )

    if getattr(eikonal3d, "inversion", "3d") == "3d":
        eikonal3d.dvp.requires_grad = True
        eikonal3d.dvs.requires_grad = True
    else:
        eikonal3d.dvp_z.requires_grad = True
        eikonal3d.dvs_z.requires_grad = True
    eikonal3d.event_loc.weight.requires_grad = False
    eikonal3d.event_time.weight.requires_grad = False

    eikonal3d = DDP(eikonal3d)
    raw_eikonal3d = eikonal3d.module

    with torch.no_grad():
        _, init_loss = eikonal3d(picks)
        init_loss_log = init_loss.detach().clone()
        dist.all_reduce(init_loss_log, op=dist.ReduceOp.SUM)
        init_loss_log = init_loss_log / ddp_world_size
    if master_process:
        print(f"Initial loss: {init_loss_log.item():.6f}", flush=True)

    if master_process:
        with open(log_path, "a") as log_file:
            log_file.write(f"inversion_mode: {inversion_mode}\n")
            log_file.write(f"lambda_dvp: {lambda_dvp}\n")
            log_file.write(f"lambda_dvs: {lambda_dvs}\n")
            log_file.write(f"lambda_sp_ratio: {lambda_sp_ratio}\n")
            log_file.write(f"beta_dvp: {beta_dvp}\n")
            log_file.write(f"beta_dvs: {beta_dvs}\n")
            log_file.write(f"beta_sp_ratio: {beta_sp_ratio}\n")
            log_file.write(f"lambda_local: {lambda_local}\n")
            log_file.write(f"n_epochs: {n_epochs}\n")
            log_file.write(f"lbfgs_max_iter: {lbfgs_max_iter}\n")
            log_file.write(f"initial_loss: {init_loss_log.item():.6f}\n")
            log_file.write(
                f"DDP station split: idx_sta % {ddp_world_size} == local_rank\n"
            )
            log_file.write(
                "Optimizing parameters:\n"
                + "\n".join(
                    f"{name}: {param.size()}"
                    for name, param in eikonal3d.named_parameters()
                    if param.requires_grad
                )
                + f"\ntime {log_timestamp()} - Inversion started (LBFGS)\n"
            )

    t0 = time.time()
    parameters = [param for param in eikonal3d.parameters() if param.requires_grad]
    optimizer = optim.LBFGS(
        params=parameters,
        max_iter=lbfgs_max_iter,
        line_search_fn="strong_wolfe",
    )

    total_closures = [0]

    for step in range(1, n_epochs + 1):
        closure_in_step = [0]

        def closure():
            optimizer.zero_grad()
            _, loss = eikonal3d(picks)
            loss.backward()

            loss_log = loss.detach().clone()
            dist.all_reduce(loss_log, op=dist.ReduceOp.SUM)
            loss_log = loss_log / ddp_world_size

            closure_in_step[0] += 1
            total_closures[0] += 1
            if master_process:
                line = (
                    f"step {step:06d} closure {closure_in_step[0]:03d} "
                    f"(total {total_closures[0]:06d}) - time {log_timestamp()} - "
                    f"L:{loss_log.item()}\n"
                )
                print(line, end="", flush=True)
                with open(log_path, "a") as log_file:
                    log_file.write(line)

            return loss_log

        loss_log = optimizer.step(closure)

        if master_process:
            vp_now, vs_now = raw_eikonal3d.current_vp_vs()
            vp_now = vp_now.detach().numpy()
            vs_now = vs_now.detach().numpy()
            np.savez(
                f"{inv_vel_path}/inv_vel_{step:06d}.npz",
                vp=vp_now,
                vs=vs_now,
                nx=nx,
                ny=ny,
                nz=nz,
                h=h,
                xgrid=xgrid,
                ygrid=ygrid,
                zgrid=zgrid,
            )
            summary = (
                f"step {step:06d} done - time {log_timestamp()} - L:{loss_log.item()} "
                f"- closures_in_step:{closure_in_step[0]}\n"
            )
            print(summary, end="", flush=True)
            with open(log_path, "a") as log_file:
                log_file.write(summary)

    with torch.no_grad():
        _, final_loss = eikonal3d(picks)
        final_loss_log = final_loss.detach().clone()
        dist.all_reduce(final_loss_log, op=dist.ReduceOp.SUM)
        final_loss_log = final_loss_log / ddp_world_size

    if master_process:
        with open(log_path, "a") as log_file:
            log_file.write(
                f"iter {n_epochs:06d} - time {log_timestamp()} - "
                f"Inversion time: {time.time() - t0:.2f}s - total_closures:{total_closures[0]}\n"
            )
            log_file.write(
                f"iter {n_epochs:06d} - time {log_timestamp()} - "
                f"Final loss:{final_loss_log.item()}\n"
            )

        vp, vs = raw_eikonal3d.current_vp_vs()
        vp = vp.detach().numpy()
        vs = vs.detach().numpy()
        np.savez(
            final_model_path,
            vp=vp,
            vs=vs,
            nx=nx,
            ny=ny,
            nz=nz,
            h=h,
            xgrid=xgrid,
            ygrid=ygrid,
            zgrid=zgrid,
        )
        with open(log_path, "a") as log_file:
            log_file.write(
                f"iter {n_epochs:06d} - time {log_timestamp()} - "
                f"Inversion results saved to {result_path}\n"
            )

    dist.destroy_process_group()
