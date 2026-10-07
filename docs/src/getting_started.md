# Getting started

## Installation
DRAGonFLy depends on `fenics-dolfinx` 0.11.0 and `mpich`, which are only distributed through conda-forge.
The `environment.yml` file in the repository root sets these up with Python 3.12 and pip-installs the package
with its remaining dependencies, including the LSDOlab packages CSDL, lsdo_geo, lsdo_function_spaces, modOpt
and IDWarp-JAX.

### For developers
```sh
$ git clone https://github.com/LSDOlab/DRAGonFLy.git
$ cd DRAGonFLy
$ conda env create -f environment.yml
$ conda activate dragonfly
```
This installs the package (`dragonfly-sim`, imported as `dragonfly_sim`) in editable mode, with the test and
documentation tools.

### For users
In an existing conda environment, install the conda-forge dependencies (skip this if they are already present),
then the package, from which pip installs the remaining dependencies:
```sh
$ conda install -c conda-forge fenics-dolfinx=0.11.0 mpich pyvista networkx
$ pip install git+https://github.com/LSDOlab/DRAGonFLy.git
```

### Dependencies
- From conda-forge: `fenics-dolfinx` 0.11.0 (not on PyPI), which brings `mpi4py`, `petsc4py`, `ufl` and `basix`,
  `mpich` (not on PyPI), `pyvista` and `networkx`.
- From PyPI, installed by pip: `numpy`, `scipy`, `matplotlib`, `jax`.
- From the `main` branches of the LSDOlab repositories, installed by pip: CSDL_alpha, lsdo_function_spaces,
  lsdo_geo, modopt and IDWarp-JAX.

All runtime dependencies are declared in `setup.py`; `pyvista` and `networkx` are listed there too, so pip would
fetch them from PyPI if they were missing.

## Running an example
From the repository root:
```sh
$ OMP_NUM_THREADS=1 mpirun -n 4 python examples/airfoil_opt.py
```
This builds the 2D airfoil optimization model and verifies its derivatives against finite differences.
[Examples](examples.md) lists the other cases, including forward and unsteady analyses and the ONERA OAT15A
validation case. Output files are written to the working directory.

`OMP_NUM_THREADS=1` matters: the conda-forge PETSc links a multithreaded BLAS, which otherwise spawns one
thread per core in every MPI rank and stalls the linear solves.

## Running the tests
```sh
$ OMP_NUM_THREADS=1 pytest
```
The tests in `tests/` cover the boundary/interior integration measures, the shape parameterization and the
shape design-variable layers, the default Euler discretization against a stored reference
(`tests/data/euler_reference.npz`), the RANS model (closures, Green-Gauss reconstruction, exact Jacobian and
adjoint derivatives against finite differences, a manufactured-solution convergence rate), time integration,
checkpoints, the far fields, forward analysis through `DG_windtunnel_model`, the VTX file writer and the mesh
input. They build their meshes in-process and take about a minute, most of it the first just-in-time compilation
of the forms. The CGNS test on the OAT15A grid is skipped when the grid is not in `meshes/`.
Some tests re-run their files under `mpirun -n 3`, so `mpirun` and `pytest` must be available in the same
environment; without `mpirun` those tests are skipped.

## Building the documentation
```sh
$ cd docs
$ make html
```
and open `docs/_build/html/index.html`. `pip install -r requirements.txt` installs the pinned documentation
dependencies that the website build uses; `pip install .[docs]` installs unpinned versions.
The website, <https://lsdolab.github.io/DRAGonFLy>, is rebuilt from `main` by GitHub Actions.
