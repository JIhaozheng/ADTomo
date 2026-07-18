import os
import subprocess

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


# Download Eigen library
def download_eigen(eigen_dir="./adtomo/eigen"):
    eigen_zip_url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    if not os.path.exists(eigen_dir):
        os.makedirs(eigen_dir, exist_ok=True)
        subprocess.check_call(["wget", "-O", "eigen.zip", eigen_zip_url])
        subprocess.check_call(["unzip", "eigen.zip", "-d", eigen_dir])
        eigen_unzipped_dir_name = os.listdir(eigen_dir)[0]
        for filename in os.listdir(os.path.join(eigen_dir, eigen_unzipped_dir_name)):
            os.rename(os.path.join(eigen_dir, eigen_unzipped_dir_name, filename), os.path.join(eigen_dir, filename))
        os.rmdir(os.path.join(eigen_dir, eigen_unzipped_dir_name))  # Clean up the unzipped directory


# Download LibTorch ## To Update
def download_libtorch(libtorch_dir="./adtomo/libtorch"):
    libtorch_zip_url = "https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip"
    if not os.path.exists(libtorch_dir):
        os.makedirs(libtorch_dir, exist_ok=True)
        subprocess.check_call(["wget", "-O", "libtorch.zip", libtorch_zip_url])
        subprocess.check_call(["unzip", "libtorch.zip", "-d", libtorch_dir])
        libtorch_unzipped_dir_name = os.listdir(libtorch_dir)[0]
        for filename in os.listdir(os.path.join(libtorch_dir, libtorch_unzipped_dir_name)):
            os.rename(
                os.path.join(libtorch_dir, libtorch_unzipped_dir_name, filename), os.path.join(libtorch_dir, filename)
            )
        os.rmdir(os.path.join(libtorch_dir, libtorch_unzipped_dir_name))  # Clean up the unzipped directory


download_eigen()
# download_libtorch()
torch_lib_dir = f"{os.path.dirname(torch.__file__)}/lib"

# Absolute include dirs: torch/ninja compiles in a temporary build dir, so relative paths like
# "./adtomo/eigen" do not resolve. Use absolute paths so every extension finds Eigen headers.
INCLUDE_DIRS = [
    os.path.abspath("./adtomo/eigen"),
    os.path.abspath("./adtomo/libtorch"),
]


def cpp_ext(name, sources):
    return CppExtension(
        name=name,
        sources=sources,
        include_dirs=INCLUDE_DIRS,
        extra_link_args=[f"-Wl,-rpath,{torch_lib_dir}"],
        language="c++",
    )


setup(
    name="adtomo",
    version="0.1.0",
    packages=["adtomo"],
    ext_modules=[
        # Discrete adjoint (forked from https://github.com/AI4EPS/ADTomo.git)
        cpp_ext("eikonal2d_op", ["adtomo/eikonal/Eikonal2D.cpp"]),
        cpp_ext("eikonal3d_op", ["adtomo/eikonal/Eikonal3D.cpp"]),
        # Continuous FSM adjoint, zero-flux (Neumann B.C.); no source-corner overwrite
        cpp_ext("eikonal2d_adj_0f_op", ["adtomo/eikonal/Eikonal2D_adj_0f.cpp"]),
        cpp_ext("eikonal3d_adj_0f_op", ["adtomo/eikonal/Eikonal3D_adj_0f.cpp"]),
        # Continuous FSM adjoint, zero-flux (Neumann B.C.) + source gradient correction
        cpp_ext("eikonal2d_adj_0f_src_op", ["adtomo/eikonal/Eikonal2D_adj_0f_src.cpp"]),
        cpp_ext("eikonal3d_adj_0f_src_op", ["adtomo/eikonal/Eikonal3D_adj_0f_src.cpp"]),
        # Continuous FSM adjoint, Dirichlet B.C.
        cpp_ext("eikonal2d_adj_dir_op", ["adtomo/eikonal/Eikonal2D_adj_dir.cpp"]),
        cpp_ext("eikonal3d_adj_dir_op", ["adtomo/eikonal/Eikonal3D_adj_dir.cpp"]),
        # Discrete SparseLU, deleted source gradient correction
        cpp_ext("eikonal2d_adj_global_op", ["adtomo/eikonal/Eikonal2D_adj_global.cpp"]),
        cpp_ext("eikonal3d_adj_global_op", ["adtomo/eikonal/Eikonal3D_adj_global.cpp"]),
        # Discrete adjoint (Li et al., 2013)
        cpp_ext("eikonal2d_adj_ordered_op", ["adtomo/eikonal/Eikonal2D_adj_ordered.cpp"]),
        cpp_ext("eikonal3d_adj_ordered_op", ["adtomo/eikonal/Eikonal3D_adj_ordered.cpp"]),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_cuda=True)},
    install_requires=[
        "torch",
    ],
    python_requires=">=3.6",
)
