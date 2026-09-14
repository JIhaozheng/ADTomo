import os
import subprocess

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = os.path.dirname(os.path.abspath(__file__))


def download_eigen(eigen_dir=os.path.join(ROOT, "adtomo", "eigen")):
    if os.path.exists(os.path.join(eigen_dir, "Eigen")):
        return
    url = "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"

    os.makedirs(eigen_dir, exist_ok=True)

    archive = os.path.join(eigen_dir, "eigen.zip")
    subprocess.check_call(["wget", "-O", archive, url])
    subprocess.check_call(["unzip", archive, "-d", eigen_dir])

    src = os.path.join(eigen_dir, "eigen-3.4.0")
    for name in os.listdir(src):
        os.rename(os.path.join(src, name), os.path.join(eigen_dir, name))

    os.rmdir(src)
    os.remove(archive)

download_eigen()

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")

def cpp_ext(name, source):
    return CppExtension(
        name=name,
        sources=[source],
        include_dirs=[os.path.join(ROOT, "adtomo", "eigen")],
        extra_link_args=[f"-Wl,-rpath,{torch_lib_dir}"],
        language="c++",
    )

setup(
    name="adtomo",
    version="0.1.0",
    packages=["adtomo"],
    ext_modules=[
        cpp_ext("eikonal2d_op", os.path.join(ROOT, "adtomo/eikonal/Eikonal2D.cpp")),
        cpp_ext("eikonal3d_op", os.path.join(ROOT, "adtomo/eikonal/Eikonal3D.cpp")),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_cuda=True)},
    install_requires=[
        "torch",
    ],
    python_requires=">=3.9",
)
