import numpy as np
import pandas as pd
import os

## generate the station file
if __name__ == "__main__":
    np.random.seed(0)

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

    num_station = 50
    reference_time = pd.to_datetime("2021-01-01T00:00:00.000")

    stations=[]
    for i in range(num_station):
        x=np.random.uniform(xgrid[0],xgrid[-1])
        y=np.random.uniform(ygrid[0],ygrid[-1])
        z=np.random.uniform(zgrid[0],zgrid[0]+3*h)
        stations.append(
            {"station_id":f"STA{i:02d}", "x_km":x,"y_km":y,"z_km":z,"dt_s":0.0})
    stations=pd.DataFrame(stations)
    stations["station_index"]=stations.index
    stations.to_csv(f"{data_path}/stations.csv", index=False)