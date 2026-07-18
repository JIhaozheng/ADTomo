import numpy as np
import pandas as pd
import os

## generate the event file
if __name__ == "__main__":
    np.random.seed(42)

    data_path = "checkerboard"
    model_path = f"{data_path}/checkerboard_true.npz"
    if not os.path.exists(data_path):
        os.makedirs(data_path)   
    meta=np.load(model_path)
    h=meta["h"]
    nx,ny,nz=meta["vp"].shape
    xgrid=meta["xgrid"]
    ygrid=meta["ygrid"]
    zgrid=meta["zgrid"]

    num_event = 100
    reference_time = pd.to_datetime("2021-01-01T00:00:00.000")

    events=[]
    for i in range(num_event):
        x=np.random.uniform(xgrid[0],xgrid[-1])
        y=np.random.uniform(ygrid[0],ygrid[-1])
        z=np.random.uniform(zgrid[0],zgrid[-1])
        t=i*5
        events.append(
            {"event_id": i, "event_time": reference_time+pd.Timedelta(seconds=t),"x_km":x,"y_km":y,"z_km":z})
    events=pd.DataFrame(events)
    events["event_index"]=events.index
    events["event_time"]=events["event_time"].apply(lambda x: x.isoformat(timespec="milliseconds"))
    events.to_csv(f"{data_path}/events.csv", index=False)