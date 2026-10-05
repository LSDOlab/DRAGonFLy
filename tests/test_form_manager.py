"""
pytest suite for form_manager.FormManager.

What these tests exist to catch is a class of SILENT failure, not a crash.
A tagged measure with no backing MeshTags assembles to exactly 0.0, so a
residual can silently become zero; test_tagged_measures_match_default_measures
is the guard for that.

MPI
---
Every test is communicator-agnostic: it takes MPI.COMM_WORLD and reduces its
assertions over owned entities only, so the file runs on any rank count:

    PYTHONNOUSERSITE=1 python -m pytest tests/test_form_manager.py -v
    mpirun -n 3 python -m pytest tests/test_form_manager.py -q \
        -p no:cacheprovider -k "not runs_under_multiple_mpi_ranks"

Every collective must be called by every rank, and reference data must be
identical on every rank.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import ufl
import dolfinx
from mpi4py import MPI

from dragonfly_sim.core.mesh_manager import Mesh
from dragonfly_sim.core.Euler_model import CompressibleEulerModel
from dragonfly_sim.utils.Euler_utils import flux
from dragonfly_sim.core.form_manager import CELL_TAG, INTERIOR_FACET_TAG

_MPI_CHILD_ENV = "FORM_MANAGER_TEST_MPI_CHILD"

WALL_TAG = 2

# Quadrature degree used for every measure here. Fixed rather than derived so
# the tagged and untagged reference forms are compiled identically -- a
# difference in quadrature would show up as a spurious mismatch.
QUADRATURE_DEGREE = 5


def build_model(poly_order=0, n=8, comm=None):
    """
    A small Euler model on a jittered unit square, with all four sides
    tagged as slip walls so the exterior-facet integrals are exercised too.

    The jitter matters because on a perfectly uniform quad mesh every
    edge of every cell ties exactly, which puts MinCellEdgeLength on a kink
    of min(). Nothing here differentiates through it, but keeping the meshes
    consistent across the suite avoids a surprise later.
    """
    comm = comm if comm is not None else MPI.COMM_WORLD
    msh = dolfinx.mesh.create_unit_square(
        comm, n, n, dolfinx.mesh.CellType.quadrilateral)
    rng = np.random.default_rng(3)
    msh.geometry.x[:, :2] += 0.01 * rng.standard_normal(
        (msh.geometry.x.shape[0], 2))

    mesh_obj = Mesh(msh, lambda pts: np.zeros(pts.shape[1], dtype=bool))

    metadata = {"quadrature_degree": QUADRATURE_DEGREE}
    mesh_obj.tag_boundary_facets(
        [(np.arange(mesh_obj.bdry_facets.size, dtype=np.int32), WALL_TAG)])
    mesh_obj.form_manager.build_boundary_measure(measure_metadata=metadata)
    mesh_obj.form_manager.build_cell_and_interior_facet_measures(
        measure_metadata=metadata)

    model = CompressibleEulerModel(mesh_obj, poly_order=poly_order)
    model.define_elements_functionspaces()
    model.define_trial_testfunctions()
    # Populates model.bcs with the weak slip-wall flux integral on ds(WALL_TAG),
    # so the residual also carries an exterior-facet integral.
    model.define_slipwall_bc(WALL_TAG)
    seed_state(model)
    return model


def seed_state(model, seed=0):
    """A smooth, strictly physical state -- no vacuum, no negative pressure."""
    V = model.functionspaces["V"]
    coords = V.tabulate_dof_coordinates()
    bs = V.dofmap.index_map_bs

    x, y = coords[:, 0], coords[:, 1]
    rho = 1.0 + 0.3 * np.sin(3.0 * x) * np.cos(2.0 * y)
    ux = 0.4 + 0.2 * np.cos(2.0 * x + y)
    uy = -0.15 + 0.1 * np.sin(x - 3.0 * y)
    p = 1.0 + 0.25 * np.cos(x + 2.0 * y)

    arr = np.zeros((coords.shape[0], bs))
    arr[:, 0] = rho
    arr[:, 1] = rho * ux
    arr[:, 2] = rho * uy
    arr[:, 3] = p / (model.gamma - 1.0) + 0.5 * rho * (ux ** 2 + uy ** 2)
    model.u_vec.x.array[:] = arr.flatten()
    model.u_vec.x.scatter_forward()


def untagged_residual(model):
    """
    The residual rebuilt on plain, untagged measures -- the pre-FormManager
    reference these tests compare against.
    """
    msh = model.mesh_obj.mesh
    metadata = {"quadrature_degree": QUADRATURE_DEGREE}
    dx = ufl.Measure("dx", domain=msh, metadata=metadata)
    dS = ufl.Measure("dS", domain=msh, metadata=metadata)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=model.mesh_obj.meshtags,
                     metadata=metadata)

    u, v, n = model.u_vec, model.v_vec, model.mesh_obj.n
    F = (-ufl.inner(ufl.grad(v), flux(u, model.gamma)) * dx
         + ufl.inner(model.flux_function(u('+'), u('-'), n('+'), model.gamma),
                     v('+') - v('-')) * dS)

    from dragonfly_sim.utils.Euler_utils import boundary_flux, slip_wall_state
    F += ufl.inner(boundary_flux(u, slip_wall_state(u, n), n, model.gamma),
                   v) * ds(WALL_TAG)
    return F


def assemble_vec(form_ufl_or_form):
    form = (form_ufl_or_form if isinstance(form_ufl_or_form, dolfinx.fem.Form)
            else dolfinx.fem.form(form_ufl_or_form))
    b = dolfinx.fem.assemble_vector(form)
    b.scatter_reverse(dolfinx.la.InsertMode.add)
    return b


def assemble_mat(form_ufl_or_form):
    form = (form_ufl_or_form if isinstance(form_ufl_or_form, dolfinx.fem.Form)
            else dolfinx.fem.form(form_ufl_or_form))
    A = dolfinx.fem.assemble_matrix(form)
    A.scatter_reverse()
    return A


def owned_slice(model, array):
    """The owned block of a DOF-indexed array, so parallel runs don't
    double-count ghosts in a reduction."""
    V = model.functionspaces["V"]
    n_owned = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    return array[:n_owned]


def global_sum(comm, value):
    return comm.allreduce(float(value), op=MPI.SUM)


class TestMeasures:

    def test_measures_carry_explicit_subdomain_ids(self):
        """
        The load-bearing precondition for L2 restriction. create_form applies
        entity lists only to integrals with a non-default subdomain id, so a
        regression to -1 here disables hyper-reduction silently.
        """
        model = build_model()
        assert model.forms.dx.subdomain_id() == CELL_TAG
        assert model.forms.dS.subdomain_id() == INTERIOR_FACET_TAG

    def test_tagged_measures_have_backing_meshtags(self):
        """A tagged measure with no subdomain_data assembles to exactly 0.0."""
        model = build_model()
        assert model.forms.dx.subdomain_data() is not None
        assert model.forms.dS.subdomain_data() is not None
        assert model.mesh_obj.meshtags is not None

    def test_tagged_measures_match_default_measures(self):
        """
        Bit-for-bit, not approximately. The all-ones MeshTags tagging is meant
        to be a pure relabelling of the same integration domain, so anything
        short of exact equality means it changed the quadrature or dropped
        entities (ghost facets are the usual casualty in parallel).
        """
        model = build_model()
        comm = model.mesh_obj.mesh.comm

        tagged = model.forms.build_residual(U_coeff=None)
        reference = untagged_residual(model)

        r_tagged = global_sum(comm, np.linalg.norm(
            owned_slice(model, assemble_vec(tagged).array)) ** 2)
        r_ref = global_sum(comm, np.linalg.norm(
            owned_slice(model, assemble_vec(reference).array)) ** 2)
        assert r_tagged == r_ref

        j_tagged = global_sum(comm, np.abs(assemble_mat(
            ufl.derivative(tagged, model.u_vec)).to_scipy().data).sum())
        j_ref = global_sum(comm, np.abs(assemble_mat(
            ufl.derivative(reference, model.u_vec)).to_scipy().data).sum())
        assert j_tagged == j_ref


@pytest.mark.skipif(shutil.which("mpirun") is None,
                    reason="mpirun is not available")
@pytest.mark.skipif(os.environ.get(_MPI_CHILD_ENV) == "1",
                    reason="already running inside the spawned MPI child")
def test_runs_under_multiple_mpi_ranks():
    """Re-run this file under mpirun, so a serial pytest invocation still
    covers the distributed path -- where the ghost-entity handling in the
    all-ones tagging is what actually gets exercised."""
    cmd = [
        "mpirun", "-n", "3",
        sys.executable, "-m", "pytest", "-x", "-q",
        "-p", "no:cacheprovider",
        "-p", "no:cov",
        __file__,
        "-k", "not runs_under_multiple_mpi_ranks",
    ]
    env = dict(os.environ)
    env[_MPI_CHILD_ENV] = "1"
    # The ranks must not inherit the parent's pytest-cov subprocess hooks: three
    # ranks tracing and writing coverage data at once is slow and can stall
    for key in [k for k in env if k.startswith(("COV_CORE_", "COVERAGE_"))]:
        del env[key]

    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=900)
    except subprocess.TimeoutExpired:
        pytest.fail(
            "the 3-rank run did not finish within 900s, which usually means "
            "one rank failed an assertion and the rest are blocked in a "
            "collective")

    assert proc.returncode == 0, (
        "3-rank run failed:\n{}\n{}".format(proc.stdout, proc.stderr))
