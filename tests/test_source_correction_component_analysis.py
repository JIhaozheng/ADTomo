"""Component-switch contract test for the production 3-D source correction."""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.source_correction_analysis.loader import load_extension

op=load_extension(); f=torch.full((9,9,9),1/6.,dtype=torch.float64); source=(3.4,3.5,3.6)
u=op.forward(f,1.,*source); gu=torch.zeros_like(f);gu[7,7,7]=1
legacy_full=op.backward(gu,u,f,1.,*source)
import eikonal3d_op
production_u=eikonal3d_op.forward(f,1.,*source)
production_g=eikonal3d_op.backward(gu,production_u,f,1.,*source)
assert torch.equal(u,production_u)
explicit=op.backward_components(gu,u,f,1.,*source,False,True,True)
assert torch.equal(explicit,production_g)
switched=op.backward_components(gu,u,f,1.,*source,True,True,True)
assert torch.equal(legacy_full,switched)
assert torch.equal(legacy_full,production_g)
for flags in ((False,False,False),(True,False,False),(False,False,True),(True,False,True),(False,True,True)):
    assert torch.isfinite(op.backward_components(gu,u,f,1.,*source,*flags)).all()
try:
    op.backward_components(gu,u,f,1.,*source,False,True,False)
except RuntimeError:
    pass
else:
    raise AssertionError("LOCAL_LU without SOURCE_GRADIENT must be rejected")
print("test_source_correction_component_analysis.py: passed")
