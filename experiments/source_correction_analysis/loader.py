"""Build the isolated component-switch extension."""

from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / "cpp/Eikonal3D_component_switches.cpp"
PRODUCTION = ROOT / "adtomo/eikonal/Eikonal3D.cpp"


def load_extension(verbose=False):
    return load(
        name="eikonal3d_source_component_analysis_op",
        sources=[str(SOURCE)],
        extra_include_paths=[str(ROOT / "adtomo/eigen")],
        extra_cflags=["-O3"],
        verbose=verbose,
    )
