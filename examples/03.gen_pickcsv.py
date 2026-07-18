## README: Use this script to generate picks file
import json
import os
import numpy as np
import pandas as pd
import torch

from adtomo.eikonal3d import eikonal3d_op, interp3d

## generate the picks file
if __name__ == "__main__":
    np.random.seed(42)
    data_path = "checkerboard"
    events=pd.read_csv(f"{data_path}/events.csv",dtype={"event_id":str})
    events["event_time"]=pd.to_datetime(events["event_time"])
    stations=pd.read_csv(f"{data_path}/stations.csv",dtype={"station_id":str})
    meta=np.load(f"{data_path}/checkerboard_true.npz")
    pick_path=f"{data_path}/picks.csv"

    h=2.0
    nx,ny,nz=meta["vp"].shape
    xgrid=np.arange(nx)*h
    ygrid=np.arange(ny)*h
    zgrid=np.arange(nz)*h
    
    eikonal_config={"nx":int(nx),"ny":int(ny),"nz":int(nz),"h":h}
    with open(f"{data_path}/eikonal_config.json", "w") as f:
        json.dump(eikonal_config, f, indent=2)
    eikonal_config.update({"xgrid":xgrid,"ygrid":ygrid,"zgrid":zgrid})

    vp=meta["vp"]
    vs=meta["vs"]
    vp=torch.from_numpy(vp)
    vs=torch.from_numpy(vs)
    events = events[(events["x_km"] >= 0) & (events["x_km"] <= (nx-2)*h) &
                    (events["y_km"] >= 0) & (events["y_km"] <= (ny-2)*h) &
                    (events["z_km"] >= 0) & (events["z_km"] <= nz*h)]
    stations = stations[(stations["x_km"] >= 0) & (stations["x_km"] <= (nx-2)*h) &
                        (stations["y_km"] >= 0) & (stations["y_km"] <= (ny-2)*h) &
                        (stations["z_km"] >= 0) & (stations["z_km"] <= nz*h)]
    # stations["z_km"] = 1.0  # with same depth as picktest.py


    picks=[]
    for _,station in stations.iterrows():
        print(f"Processing station {station['station_id']}")
        tp3d=eikonal3d_op.forward(1.0/vp, h, 
                                    station["x_km"]/h, 
                                    station["y_km"]/h, 
                                    station["z_km"]/h).numpy()
        ts3d=eikonal3d_op.forward(1.0/vs, h, 
                                    station["x_km"]/h,
                                    station["y_km"]/h,
                                    station["z_km"]/h).numpy()
        for _,event in events.iterrows():
            # if np.random.rand() < 0.5:
            if 1==1:
                tt=interp3d(tp3d, event["x_km"],event["y_km"],event["z_km"],
                    eikonal_config["xgrid"],eikonal_config["ygrid"],eikonal_config["zgrid"],h)
                picks.append({
                    "event_id": event["event_id"],
                    "station_id": station["station_id"],
                    "phase_type": "P",
                    "phase_time": event["event_time"]+pd.Timedelta(seconds=tt),
                    "travel_time":tt,
                })
            # if np.random.rand() < 0.3:
            if 1==1:
                tt=interp3d(ts3d, event["x_km"],event["y_km"],event["z_km"],
                    eikonal_config["xgrid"],eikonal_config["ygrid"],eikonal_config["zgrid"],h)
                picks.append({
                    "event_id": event["event_id"],
                    "station_id": station["station_id"],
                    "phase_type": "S",
                    "phase_time": event["event_time"]+pd.Timedelta(seconds=tt),
                    "travel_time":tt,
                })
    picks=pd.DataFrame(picks)
    picks["phase_time"]=picks["phase_time"].apply(lambda x:x.isoformat(timespec="milliseconds"))
    picks["event_index"]=picks["event_id"].map(events.set_index("event_id")["event_index"])
    picks["station_index"]=picks["station_id"].map(stations.set_index("station_id")["station_index"])
    picks.to_csv(f"{pick_path}", index=False)
    print(f"Generated {len(picks)} picks and saved to {pick_path}")