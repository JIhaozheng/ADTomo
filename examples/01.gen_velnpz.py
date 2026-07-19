import numpy as np
import os

## generate the velocity model file
if __name__ == "__main__":
    data_path = "checkerboard"
    data_name = "checkerboard_true.npz"
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # nx, ny, nz = 200, 645, 20
    nx, ny, nz = 20, 65, 10
    
    h = 2

    xgrid = np.arange(nx) * h
    ygrid = np.arange(ny) * h
    zgrid = np.arange(nz) * h

    x_km = xgrid[:, None, None]
    y_km = ygrid[None, :, None]
    z_km = zgrid[None, None, :]
    # background: vp = min(5 + 0.11*z, 7.8)  (z in km)
    vp_bg = np.minimum(5.0 + 0.11 * z_km, 7.8)

    # smooth checkerboard: low-frequency cos*cos in x,y, gentle sin in z (domain-normalized)
    Lx = 5*h
    Ly = 5*h
    Lz = 4*h
    amp = 0.2
    dv_v = amp * np.cos(np.pi * x_km / Lx) * np.cos(np.pi * y_km / Ly) * np.sin(
        np.pi * z_km / Lz
    )
    vp = np.broadcast_to(vp_bg * (1 + dv_v), (nx, ny, nz)).copy()
    vs = vp / 1.732

    np.savez(
        f"{data_path}/{data_name}",
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
