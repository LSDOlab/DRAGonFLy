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
from dragonfly_sim.core.form_manager import CELL_TAG, INTERIOR_FACET_TAG, EntitySet

_MPI_CHILD_ENV = "FORM_MANAGER_TEST_MPI_CHILD"

WALL_TAG = 2

# Quadrature degree used for every measure here. Fixed rather than derived so
# the tagged and untagged reference forms are compiled identically -- a
# difference in quadrature would show up as a spurious mismatch.
QUADRATURE_DEGREE = 5


pytestmark = [pytest.mark.unit_test]


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


class TestRestriction:
    """The restricted twin (FormManager.restrict) of hyper-reduction: the same compiled
    form, bound over a subset of the entities."""

    def test_restricted_all_entities_matches_fom(self):
        """restrict() over every entity reproduces dolfinx.fem.form exactly."""
        model = build_model()
        comm = model.mesh_obj.mesh.comm
        forms = model.forms
        forms.compiled("J", lambda: ufl.derivative(forms.build_residual(), model.u_vec))
        twin = forms.restrict(entity_set=forms.entity_set_all(boundary_tags=[WALL_TAG]))
        fom = global_sum(comm, np.abs(assemble_mat(forms.compiled("J")).to_scipy().data).sum())
        restricted = global_sum(comm, np.abs(assemble_mat(twin.compiled("J")).to_scipy().data).sum())
        assert restricted == fom

    def test_restricted_twin_shares_the_parent_kernel(self):
        """No re-JIT: the twin binds the parent's UFL expression and kernel."""
        model = build_model()
        forms = model.forms
        forms.compiled("F_physical", lambda: forms.build_residual())
        twin = forms.restrict(entity_set=forms.entity_set_all(boundary_tags=[WALL_TAG]))
        twin.compiled("F_physical")
        assert twin._kernels is forms._kernels
        assert twin.ufl_form("F_physical") is forms.ufl_form("F_physical")

    def test_restricted_subset_touches_only_sampled_rows(self):
        """
        The guard against ignored entity lists (create_form applies them only to
        integrals with a non-default subdomain id): with the integrals over a subset of
        the interior facets alone, only the rows of the cells next to them are written.
        (Facets: at p = 0 the cell integral of the residual vanishes, grad v = 0.)
        """
        model = build_model()
        forms = model.forms
        msh = model.mesh_obj.mesh
        tdim = msh.topology.dim
        forms.compiled("F_physical", lambda: forms.build_residual())
        facets = _owned_interior_facets(msh)[::5]
        twin = forms.restrict(entity_set=EntitySet(
            msh, cells=np.zeros(0, dtype=np.int32), interior_facets=facets, exterior_facets_by_tag={}))
        b = assemble_vec(twin.compiled("F_physical")).array
        touched = np.nonzero(np.abs(owned_slice(model, b)) > 0.0)[0]
        # the owned cells next to a sampled facet of any rank (a facet owned by another rank
        # reaches this rank's rows through the ghost accumulation)
        f_to_c = msh.topology.connectivity(tdim - 1, tdim)
        marker = dolfinx.la.vector(msh.topology.index_map(tdim), 1, dtype=np.int32)
        marker.array[:] = 0
        for f in facets:
            marker.array[f_to_c.links(int(f))] = 1
        marker.scatter_reverse(dolfinx.la.InsertMode.add)
        cells = np.flatnonzero(marker.array[:msh.topology.index_map(tdim).size_local]).astype(np.int32)
        assert np.all(np.isin(touched, forms.dofs_for_cells(cells)))
        comm = msh.comm
        assert 0 < global_sum(comm, touched.size) < global_sum(comm, owned_slice(model, b).size)

    def test_rebind_changes_the_sample_set(self):
        model = build_model()
        forms = model.forms
        msh = model.mesh_obj.mesh
        forms.compiled("F_physical", lambda: forms.build_residual())
        facets = _owned_interior_facets(msh)
        no_cells = np.zeros(0, dtype=np.int32)
        twin = forms.restrict(entity_set=EntitySet(
            msh, cells=no_cells, interior_facets=facets[::5], exterior_facets_by_tag={}))
        few = np.count_nonzero(owned_slice(model, assemble_vec(twin.compiled("F_physical")).array))
        twin.rebind(EntitySet(msh, cells=no_cells, interior_facets=facets, exterior_facets_by_tag={}))
        many = np.count_nonzero(owned_slice(model, assemble_vec(twin.compiled("F_physical")).array))
        assert 0 < global_sum(msh.comm, few) < global_sum(msh.comm, many)


def _owned_interior_facets(msh):
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = msh.topology.connectivity(tdim - 1, tdim)
    n_owned = msh.topology.index_map(tdim - 1).size_local
    return np.array([f for f in range(n_owned) if len(f_to_c.links(f)) == 2], dtype=np.int32)


class TestDofEntityBridge:

    def test_cells_for_dofs_inverts_dofs_for_cells(self):
        model = build_model()
        forms = model.forms
        msh = model.mesh_obj.mesh
        n_owned_cells = msh.topology.index_map(msh.topology.dim).size_local
        cells = np.arange(0, n_owned_cells, 3, dtype=np.int32)
        assert np.array_equal(np.sort(forms.cells_for_dofs(forms.dofs_for_cells(cells))), np.sort(cells))

    def test_restricted_assembly_reproduces_sampled_rows_exactly(self):
        """
        entity_set_for_cells names every entity contributing to the sampled cells' rows
        (their volume integral and every facet they touch, also facets owned by another
        rank), so the restricted assembly's sampled rows equal the full ones exactly.
        """
        model = build_model()
        forms = model.forms
        msh = model.mesh_obj.mesh
        n_owned_cells = msh.topology.index_map(msh.topology.dim).size_local
        forms.compiled("F_physical", lambda: forms.build_residual())
        fom = assemble_vec(forms.compiled("F_physical")).array
        cells = np.arange(0, n_owned_cells, 4, dtype=np.int32)
        twin = forms.restrict(entity_set=forms.entity_set_for_cells(cells))
        restricted = assemble_vec(twin.compiled("F_physical")).array
        rows = forms.dofs_for_cells(cells)
        rows = rows[rows < owned_slice(model, fom).size]
        assert np.array_equal(fom[rows], restricted[rows])
        # a restriction, not a full assembly: fewer rows touched
        comm = msh.comm
        assert global_sum(comm, np.count_nonzero(owned_slice(model, restricted))) < \
            global_sum(comm, np.count_nonzero(owned_slice(model, fom)))

    def test_cell_neighbours_is_the_facet_ring(self):
        """cell_neighbours adds every facet neighbour, also across ranks: the global count
        matches a serial computation on the global cell indices."""
        model = build_model()
        forms = model.forms
        msh = model.mesh_obj.mesh
        tdim = msh.topology.dim
        imap = msh.topology.index_map(tdim)
        comm = msh.comm
        # the owned cells whose global index is a multiple of 7, and their ring
        globals_owned = imap.local_to_global(np.arange(imap.size_local, dtype=np.int32))
        cells = np.flatnonzero(globals_owned % 7 == 0).astype(np.int32)
        ring = forms.cell_neighbours(cells)
        assert np.all(np.isin(cells, ring))
        ring_global = np.sort(np.concatenate(comm.allgather(globals_owned[ring])))
        # serial reference over the facet adjacency of all (owned and ghost) cells
        msh.topology.create_connectivity(tdim, tdim - 1)
        msh.topology.create_connectivity(tdim - 1, tdim)
        c_to_f, f_to_c = msh.topology.connectivity(tdim, tdim - 1), msh.topology.connectivity(tdim - 1, tdim)
        n_all = imap.size_local + imap.num_ghosts
        globals_all = imap.local_to_global(np.arange(n_all, dtype=np.int32))
        pairs = set()
        for c in range(n_all):
            for f in c_to_f.links(c):
                for d in f_to_c.links(int(f)):
                    pairs.add((int(globals_all[c]), int(globals_all[d])))
        pairs = set().union(*comm.allgather(pairs))
        sampled = {g for g in range(imap.size_global) if g % 7 == 0}
        expected = sampled | {d for c, d in pairs if c in sampled}
        assert ring_global.tolist() == sorted(expected)


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
