import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
 
    # data_path="checkerboard/inversion/inv_model.npz"
    data_path="checkerboard/inversion/inv_model.npz"
    figure_path="checkerboard/inversion/figures_inv"
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)
    model = np.load(data_path)
    h=2.0
    nx,ny,nz=model["vp"].shape
    vp=model["vp"]
    vs=model["vs"]
    vp_mean=vp.mean(axis=(0,1))
    print(f"vp_mean: {vp_mean}")
    print(f"dx: {h}, nx: {nx}, ny: {ny}, nz: {nz}")
    depth=np.arange(nz)*h
    plt.plot(vp.mean(axis=(0,1)),depth,'-',color='k',label="P_mean")
    plt.plot(vs.mean(axis=(0,1)),depth,'-',color='r',label="S_mean")
    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Depth (km)")
    plt.gca().invert_yaxis()
    plt.title(f"Velocity vs Depth")
    plt.legend()
    plt.savefig(f"{figure_path}/1dProfile.png")

    print(f"Saved 1d figure to {os.path.join(figure_path)}")
    for slice_idz in range(nz):
        depth = h * (slice_idz + 0.5)

        # Compute correct extent in physical coordinates
        x_extent = [0, nx * h]
        y_extent = [0, ny * h]
        z_extent = [0, nz * h]
        print(f"vp_diff,vp_max,vp_min: {np.max(vp[:,:,slice_idz])-np.min(vp[:,:,slice_idz])}, {np.max(vp[:,:,slice_idz])}, {np.min(vp[:,:,slice_idz])}")
        print(f"vs_diff,vs_max,vs_min: {np.max(vs[:,:,slice_idz])-np.min(vs[:,:,slice_idz])}, {np.max(vs[:,:,slice_idz])}, {np.min(vs[:,:,slice_idz])}")
        # Keep the aspect ratio as true physical scale: aspect='equal'
        fig, ax = plt.subplots(1, 2, figsize=(8, 8))
        vp_region=vp.mean(axis=(0,1))[slice_idz]*np.array([0.9,1.1])
        vs_region=vs.mean(axis=(0,1))[slice_idz]*np.array([0.9,1.1])

        im0 = ax[0].imshow(
            vp[:, :, slice_idz].T,
            cmap="seismic_r",
            origin="lower",
            extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
            aspect='equal',
            vmin=vp_region[0],
            vmax=vp_region[1]
        )
        ax[0].set_title(f"Vp\n(z={slice_idz}, depth={depth:.2f} km)")
        ax[0].set_xlabel('x (km)',fontsize=12)
        ax[0].set_ylabel('y (km)',fontsize=12)
        plt.colorbar(im0, ax=ax[0], fraction=0.03)

        im1 = ax[1].imshow(
            vs[:, :, slice_idz].T,
            cmap="seismic_r",
            origin="lower",
            extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]],
            aspect='equal',
            vmin=vs_region[0],
            vmax=vs_region[1]
        )
        ax[1].set_title(f"Vs\n(z={slice_idz}, depth={depth:.2f} km)",fontsize=12)
        ax[1].set_xlabel('x (km)',fontsize=12)
        ax[1].set_ylabel('y (km)',fontsize=12)
        plt.colorbar(im1, ax=ax[1], fraction=0.03)

        plt.tight_layout()
        plt.savefig(os.path.join(figure_path, f"vel_depth_{depth:.2f}km.png"))
        plt.close()

