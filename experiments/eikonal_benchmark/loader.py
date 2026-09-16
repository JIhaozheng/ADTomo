"""Load frozen benchmark solvers with identical CPU compilation flags."""
from __future__ import annotations

import hashlib
from pathlib import Path

from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CPP = HERE / "cpp"
VARIANTS = {
    "old_full": CPP / "Eikonal3D_old_full.cpp",
    "current": CPP / "Eikonal3D_current.cpp",
    "weiqiang": CPP / "Eikonal3D_weiqiang.cpp",
}
HASHES = {
    "old_full": "28277d7eb2dc99dbb3e7e84938a9021d5e00cb5e393f31ea3dc276b264b1a3f1",
    "current": "6180c0b8aeb2a15c3e9507b01c158c5e7dd282c334a80976e835b261e12f0dec",
    "weiqiang": "ddebcf14686c64cccc9a4e48fddbc885f528ab3c87e20ceb10d23e7fdc24d28f",
}


def validate_frozen_sources() -> None:
    for name, path in VARIANTS.items():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != HASHES[name]:
            raise RuntimeError(f"Frozen {name} source changed: {path}")


def load_variants(verbose: bool = False):
    validate_frozen_sources()
    return {
        name: load(
            name=f"eikonal_benchmark_{name}", sources=[str(path)],
            extra_include_paths=[str(ROOT / "adtomo" / "eigen")],
            extra_cflags=["-O3", "-DNDEBUG"], verbose=verbose,
        )
        for name, path in VARIANTS.items()
    }
