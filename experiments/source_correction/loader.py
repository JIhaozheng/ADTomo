"""Build the isolated source-correction extensions without touching setup.py."""

from __future__ import annotations

import hashlib
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CPP = HERE / "cpp"
PRODUCTION = ROOT / "adtomo/eikonal/Eikonal3D.cpp"
VARIANTS = {
    "full": ("eikonal3d_full_op", CPP / "Eikonal3D_full.cpp"),
    "no_lu": ("eikonal3d_no_lu_op", CPP / "Eikonal3D_no_lu.cpp"),
    "bulk_only": ("eikonal3d_bulk_only_op", CPP / "Eikonal3D_bulk_only.cpp"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _forward_section(text: str) -> str:
    start = text.index("static inline int gid")
    end = text.index("static inline bool is_source_corner")
    return text[start:end]


def validate_sources() -> None:
    """Fail early if a variant accidentally changes the common forward path."""
    full = VARIANTS["full"][1]
    # ``full`` is the frozen pre-refactor baseline used for the original
    # controlled experiment.  Production now expresses the same forward with
    # SourceCell, so enforce byte identity among experiment variants and check
    # production equality numerically in the focused test instead.
    reference = _forward_section(full.read_text(encoding="utf-8"))
    for name, (_, source) in VARIANTS.items():
        if _forward_section(source.read_text(encoding="utf-8")) != reference:
            raise RuntimeError(f"{name} changes the shared forward implementation")

    no_lu = VARIANTS["no_lu"][1].read_text(encoding="utf-8")
    no_lu_backward = no_lu[no_lu.index("static void backward("):no_lu.index("// ---------------------------------------------------------------------------\n// PyTorch bindings")]
    if "patch_lu_corner_res_3d(" in no_lu_backward:
        raise RuntimeError("no_lu still calls the local LU correction")

    bulk = VARIANTS["bulk_only"][1].read_text(encoding="utf-8")
    bulk_backward = bulk[bulk.index("static void backward("):bulk.index("// ---------------------------------------------------------------------------\n// PyTorch bindings")]
    if "patch_lu_corner_res_3d(" in bulk_backward or "apply_source_simpson_grad_from_res(" in bulk_backward:
        raise RuntimeError("bulk_only still applies a source correction")
    if "z, false);" not in bulk_backward:
        raise RuntimeError("bulk_only does not disable the source-aware adjoint stencil")


def load_extensions(verbose: bool = False):
    """Return ``{full, no_lu, bulk_only}`` modules built with identical flags."""
    validate_sources()
    eigen = ROOT / "adtomo/eigen"
    modules = {}
    for variant, (module_name, source) in VARIANTS.items():
        modules[variant] = load(
            name=module_name,
            sources=[str(source)],
            extra_include_paths=[str(eigen)],
            extra_cflags=["-O3"],
            verbose=verbose,
        )
    return modules
