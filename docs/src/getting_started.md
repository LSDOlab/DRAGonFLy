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
$ git lfs install
$ git lfs pull
```
This installs the package (`dragonfly-sim`, imported as `dragonfly_sim`) in editable mode, with the test and
documentation tools. The meshes in `meshes/` are stored with Git LFS; `git lfs pull` fetches them.

### For users
In a conda environment that provides `fenics-dolfinx` 0.11.0 and `mpich` (for example the one above), run
```sh
$ pip install git+https://github.com/LSDOlab/DRAGonFLy.git
```

## Running an example
From the repository root:
```sh
$ OMP_NUM_THREADS=1 mpirun -n 4 python examples/airfoil_opt.py
```
This builds the 2D airfoil optimization model and verifies its derivatives against finite differences
(see [Examples](examples.md)). Output files are written to the working directory.

`OMP_NUM_THREADS=1` matters: the conda-forge PETSc links a multithreaded BLAS, which otherwise spawns one
thread per core in every MPI rank and stalls the linear solves.

## Running the tests
```sh
$ OMP_NUM_THREADS=1 pytest
```
The tests in `tests/` cover the boundary/interior integration measures and the shape parameterization.
One test re-runs its file under `mpirun -n 3`, so `mpirun` and `pytest` must be available in the same
environment; without `mpirun` that test is skipped.

## Building the documentation
```sh
$ cd docs
$ make html
```
and open `docs/_build/html/index.html`. `pip install -r requirements.txt` installs the pinned documentation
dependencies that the website build uses; `pip install .[docs]` installs unpinned versions.
The website, <https://lsdolab.github.io/DRAGonFLy>, is rebuilt from `main` by GitHub Actions.
