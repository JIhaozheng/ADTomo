# ADTomo

3D traveltime tomography with PyTorch (C++ Eikonal / adjoint extensions).

## 1. Environment (conda recommended)

```bash
conda create -n adtomo python=3.9 -y
conda activate adtomo

conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install numpy pandas matplotlib ninja pybind11 setuptools wheel -y
```

CPU-only: replace the PyTorch line with  
`conda install pytorch torchvision cpuonly -c pytorch -y`.

## 2. Build and install

```bash
cd /path/to/ADTomo
python setup.py build_ext --inplace
pip install -e . --no-build-isolation
```

Check: `python -c "from adtomo.Eikonal3d_0f_src import Eikonal3D; print('ok')"`.

## 3. Backends (short names)

| Name | Meaning |
|------|---------|
| **FSM** | Fast Sweeping Method |
| **`LU`** | Discrete adjoint (forked from [AI4EPS/ADTomo](https://github.com/AI4EPS/ADTomo)); gradient baseline |
| **`global`** | SparseLU discrete adjoint |
| **`0f`** | Continuous FSM + Neumann (zero-flux) B.C. |
| **`0f_src`** | Same as `0f` + source gradient correction |
| **`dir`** | Continuous FSM + Dirichlet B.C. |
| **`ordered`** | Discrete ordered adjoint (Li et al., 2013) |

## 4. Tests

```bash
cd tests
python compare_with_LUbaseline.py   # vs LU: timing + gradient fields
python test_all_eikonal_grad.py     # gradient test for all adjoint backends
```

Figures: `tests/gradtest_figs/`.

`LU`, `ordered`, and `0f_src` pass the gradient test; vs LU, they also have the smallest gradient errors. Neumann B.C. (`0f`) and Dirichlet B.C. (`dir`) can reach ~1e-1 error at a few points; `dir` can reach 
~1e-8 near the boundary.

<img height="400" alt="gradient_field" src="https://github.com/user-attachments/assets/339e5086-72f7-440b-bd92-2651a18feaa0" /><img height="400" alt="gradient_test" src="https://github.com/user-attachments/assets/27a77131-b444-49ff-b126-09f4f951de8c" />



## 5. Checkerboard example

```bash
cd examples
python 01.gen_velnpz.py    # set amp for true / initial velocity
python 02.a.gen_evecsv.py
python 02.b.gen_stacsv.py
python 03.gen_pickcsv.py     # picks from true velocity
python 04.inversion.py       # single-process inversion
# torchrun --nproc_per_node=2 04.inversion_parallel.py   # optional
python 05.readnpz.py
```

Output: `examples/checkerboard/inversion/`.
<img width="600" alt="loss_curve" src="https://github.com/user-attachments/assets/d04139e1-de8b-41f9-bb12-171bcd1d9dab" />
