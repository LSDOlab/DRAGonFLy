"""
core/reduced_order_model.py: the LSPG POD reduced-order model, for the Euler
and the RANS model through the same interface, and its use in
DG_windtunnel_model.

The flow cases are the jittered unit square of test_windtunnel_forward.py
with the bottom side as the body wall: slip for Euler, adiabatic no-slip
(SA-neg RANS, pseudo-transient continuation) for RANS.
"""
import os

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx

from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.core.reduced_order_model import ReducedOrderModel, LSPGSolver
from dragonfly_sim.core.RANS_model import CompressibleRANSModel
from dragonfly_sim.core.pod import PODBasis

from cases import jittered_square

BC = {'inlet': {'rho': 1.0, 'p': 1.0, 'M': 0.5, 'alpha': np.radians(2.0)}, 'outlet': {'p': 1.0}}


_WINDTUNNELS = []


def _close_writers(wt):
    """Close a windtunnel's ADIOS2 writers. A writer destroyed later, after the
    working directory changed, cannot find its relative path and ADIOS2
    aborts the process."""
    m, mesh = wt.sim_model, wt.mesh
    writers = [getattr(m, "fom_solution_writer", None), getattr(m, "fom_pressure_writer", None),
               getattr(mesh, "_deformation_writer", None)]
    writers += list((getattr(mesh, "_quality_writers", None) or {}).values())
    for w in writers:
        if w is not None and w.writer is not None:
            w.writer.close()
            w.writer = None


@pytest.fixture
def in_tmp_dir(tmp_path):
    """The windtunnel writes output files into the working directory."""
    comm = MPI.COMM_WORLD
    path = comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0)
    cwd = os.getcwd()
    os.chdir(path)
    yield
    while _WINDTUNNELS:
        _close_writers(_WINDTUNNELS.pop())
    comm.Barrier()
    os.chdir(cwd)


def make_windtunnel(kind, rom=None, Re=200.0, suffix="rom"):
    kw = {}
    if kind == "rans":
        kw = dict(model_class=CompressibleRANSModel, model_kwargs={"Re": Re})
    wt = DG_windtunnel_model(jittered_square(8), BC, mesh_inner_bdry_function=lambda x: x[1] < 1e-8,
                             poly_order=0, filename_suffix="{}_{}".format(suffix, kind),
                             reduced_order_model=rom, **kw)
    wt.euler_residual_conv_limit = 1e-11
    wt.log_cm_alpha = False
    wt.set_up_sim()
    _WINDTUNNELS.append(wt)
    return wt


def global_array(comm, local):
    return np.concatenate(comm.allgather(np.asarray(local)))


# ----------------------------------------------------------------------
# Model interface
# ----------------------------------------------------------------------
def test_state_scales(in_tmp_dir):
    wt = make_windtunnel("euler")
    m = wt.sim_model
    inlet = m.boundary_conditions['inlet']
    U = inlet['M'] * inlet['c']
    assert m.state_scales() == pytest.approx([1.0, U, U, 1.0 / 0.4 + 0.5 * U**2], rel=1e-14)


def test_rans_state_scales_match_the_limiter_scales(in_tmp_dir):
    wt = make_windtunnel("rans", Re=6e6)
    m = wt.sim_model
    tm = m.turbulence_model
    scales = m.state_scales()
    assert len(scales) == m.n_state == 5
    expected_extra = 1e3 * tm.chi_inf * float(m.mu_inf_const.value)
    assert scales[4] == pytest.approx(expected_extra, rel=1e-12)
    # the MUSCL limiter reads the same expressions
    lim = m._limiter_scales()
    assert [float(s) for s in lim[:4]] == pytest.approx(scales[:4], rel=1e-14)


def test_steady_residual_switches_the_pseudo_time_term_off(in_tmp_dir):
    wt = make_windtunnel("rans")
    m = wt.sim_model
    assert m.use_ptc
    m.set_up_solver()
    term = m.time_integrator.term
    term.set_cfl(3.0)
    U_old = term.U_old
    U_old.x.array[:] = m.u_vec.x.array
    U_old.x.array[0::m.n_state] *= 1.01        # make the pseudo term nonzero
    U_old.x.scatter_forward()

    b_ptc = m.problem._b.duplicate()
    m.problem.F(m.u_vec.x.petsc_vec, b_ptc)
    with m.steady_residual():
        assert float(term.inv_cfl.value) == 0.0 and float(term.inv_dt.value) == 0.0
        b_steady = m.problem._b.duplicate()
        m.problem.F(m.u_vec.x.petsc_vec, b_steady)
    assert float(term.inv_cfl.value) == pytest.approx(1.0 / 3.0)
    m._ensure_residual_vector()
    m._assemble_physical_residual()
    diff = b_steady.copy()
    diff.axpy(-1.0, m.residual_vec_physical)
    assert diff.norm() <= 1e-12 * max(m.residual_vec_physical.norm(), 1.0)
    b_ptc.axpy(-1.0, b_steady)
    assert b_ptc.norm() > 1e-6
    for v in (b_ptc, b_steady, diff):
        v.destroy()


def test_positivity_step_length_caps_and_limits(in_tmp_dir):
    m = make_windtunnel("euler").sim_model
    x = m.u_vec.x.petsc_vec
    dx = x.duplicate()
    dx.set(0.0)
    theta, cap = m.positivity_step_length(x, dx)
    assert cap == 0.0
    # a step that empties the density everywhere at theta = 1
    dx.array[0::m.n_state] = x.array[0::m.n_state]
    theta, cap = m.positivity_step_length(x, dx)
    assert 0.0 < theta < 1.0 and theta <= cap
    dx.destroy()


# ----------------------------------------------------------------------
# Inner product and basis
# ----------------------------------------------------------------------
def test_default_W_is_the_state_scaled_euclidean_product(in_tmp_dir):
    rom = ReducedOrderModel(rb_size=4)
    assert (rom.inner_product, rom.state_scaling) == ("euclidean", "freestream")
    wt = make_windtunnel("rans", rom=rom, Re=6e6)
    m = wt.sim_model
    w = rom.W.w.getArray().reshape(-1, m.n_state)
    assert w == pytest.approx(np.tile(1.0 / rom.scales**2, (w.shape[0], 1)), rel=1e-14)


def test_W_is_the_scaled_L2_inner_product(in_tmp_dir):
    rom = ReducedOrderModel(rb_size=4, inner_product="l2")
    wt = make_windtunnel("rans", rom=rom, Re=6e6)
    m = wt.sim_model
    s = rom.scales
    u = m.u_vec
    form = sum(u[k]**2 / s[k]**2 for k in range(m.n_state)) * m.forms.dx
    comm = m.mesh_obj.mesh.comm
    exact = comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), op=MPI.SUM)
    x = u.x.petsc_vec
    Wx = x.duplicate()
    rom.W.mult(x, Wx)
    assert x.dot(Wx) == pytest.approx(exact, rel=1e-12)
    Wx.destroy()


@pytest.mark.parametrize("inner_product, scaling", [("l2", "freestream"), ("l2", None), ("euclidean", None)])
def test_restriction_inverts_expansion(in_tmp_dir, inner_product, scaling):
    rom = ReducedOrderModel(rb_size=3, inner_product=inner_product, state_scaling=scaling)
    wt = make_windtunnel("euler", rom=rom)
    m = wt.sim_model
    comm = m.mesh_obj.mesh.comm
    rng = np.random.default_rng(1)
    x = m.u_vec.x.petsc_vec
    for j in range(3):
        x.array[:] = rng.standard_normal(x.getLocalSize())
        rom.pod.snapshot_matrix.process_and_append_to_snapshot_lists(x, np.array([float(j)]))
    rom.pod.construct_basis(rb_size=3, layout_vec=x)
    solver = LSPGSolver(rom.pod.basis, comm, W=None if rom.W is None else rom.W.w)
    a = np.array([0.3, -1.2, 2.0])
    solver.expand(a, m.u_vec)
    assert solver.restrict(x) == pytest.approx(a, rel=1e-12, abs=1e-12)
    assert (rom.W is None) == (inner_product == "euclidean" and scaling is None)


def test_scaled_L2_keeps_the_turbulence_variable_under_truncation(in_tmp_dir):
    """
    At Re 6e6 rho*nu_tilde is ~1e-4 against rho*E ~ 2. Four snapshots: three
    mean-flow perturbations (relative sizes 1, 1, 1e-3) and one perturbing
    rho*nu_tilde alone by its own scale; max_rank 3 drops one. The Euclidean
    POD drops the rho*nu_tilde direction; with the state scaling (with or
    without the mass matrix) the small mean-flow perturbation goes instead.
    """
    errors = {}
    for inner, scaling in (("euclidean", "freestream"), ("l2", "freestream"), ("euclidean", None)):
        rom = ReducedOrderModel(rb_size=3, max_rank=3, inner_product=inner, state_scaling=scaling)
        wt = make_windtunnel("rans", rom=rom, Re=6e6, suffix="trunc_{}_{}".format(inner, scaling))
        m = wt.sim_model
        comm = m.mesh_obj.mesh.comm
        scales = np.asarray(m.state_scales())
        x = m.u_vec.x.petsc_vec
        n_cells = x.getLocalSize() // m.n_state
        rng = np.random.default_rng(5 + comm.rank)
        snapshots = []
        for k, amp in enumerate((1.0, 1.0, 1e-3)):
            d = np.zeros((n_cells, m.n_state))
            d[:, :4] = amp * 1e-2 * scales[:4] * rng.standard_normal((n_cells, 4))
            snapshots.append(d.ravel())
        d = np.zeros((n_cells, m.n_state))
        d[:, 4] = 1e-2 * scales[4] * rng.standard_normal(n_cells)
        snapshots.append(d.ravel())
        for j, snap in enumerate(snapshots):
            x.array[:] = snap
            rom.pod.snapshot_matrix.process_and_append_to_snapshot_lists(x, np.array([float(j)]))
        rom.pod.construct_basis(rb_size=3, layout_vec=x)
        solver = LSPGSolver(rom.pod.basis, comm, W=None if rom.W is None else rom.W.w)
        # relative W-error of projecting the turbulence snapshot onto the basis
        x.array[:] = snapshots[3]
        solver.expand(solver.restrict(x), m.u_vec)
        proj = m.u_vec.x.petsc_vec.getArray().copy()
        err = comm.allreduce(np.sum(((proj - snapshots[3]).reshape(-1, m.n_state)[:, 4])**2), op=MPI.SUM)
        ref = comm.allreduce(np.sum(snapshots[3].reshape(-1, m.n_state)[:, 4]**2), op=MPI.SUM)
        errors[(inner, scaling)] = np.sqrt(err / ref)
    assert errors[("euclidean", "freestream")] < 1e-8
    assert errors[("l2", "freestream")] < 1e-8
    assert errors[("euclidean", None)] > 0.5


# ----------------------------------------------------------------------
# LSPG solve, Euler and RANS through one interface
# ----------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["euler", "rans"])
def test_lspg_recovers_a_solution_in_the_basis(in_tmp_dir, kind):
    """With the converged FOM state u* in the snapshot set, the ROM started
    from another state converges back to u* and to FOM-level eta, for each
    inner product. The three FOM solves are shared by the inner products."""
    wt = make_windtunnel(kind)
    m = wt.sim_model
    comm = m.mesh_obj.mesh.comm
    states = {}
    for a in (2.0, 2.8, 2.4):
        u, converged, _ = wt.solve_forward(np.radians(a))
        assert converged
        states[a] = u
    m.set_angle_of_attack(np.radians(2.4))
    for inner_product, scaling in (("euclidean", "freestream"), ("l2", "freestream"), ("euclidean", None)):
        rom = ReducedOrderModel(rb_size=5, inner_product=inner_product, state_scaling=scaling,
                                eta_threshold=1e-9)
        rom.set_up(m)
        for a, u in states.items():
            m.set_u_vec(u)
            rom.add_snapshot(m, np.array([np.radians(a)]))
        # start from u*(2.0) at alpha 2.4
        m.set_u_vec(states[2.0])
        if kind == "rans":
            m.time_integrator.term.set_cfl(7.0)
        result = rom.solve(m, np.array([np.radians(2.4)]))
        assert result is not None and result.converged and result.accepted, (inner_product, scaling)
        u_rom, u_fom = global_array(comm, result.solution), global_array(comm, states[2.4])
        assert np.linalg.norm(u_rom - u_fom) <= 1e-8 * np.linalg.norm(u_fom), (inner_product, scaling)
        assert result.eta < 1e-10, (inner_product, scaling)
        if kind == "rans":
            assert float(m.time_integrator.term.inv_cfl.value) == pytest.approx(1.0 / 7.0)
        rom.destroy()


# ----------------------------------------------------------------------
# DG_windtunnel_model
# ----------------------------------------------------------------------
# alpha within the 1 degree of the inlet/outlet tagging angle (2 degrees). The
# ROM interpolating between two snapshots reaches eta ~3e-7 (Euler) and is
# rejected; with three it reaches 1e-10 (Euler) / 5e-9 (RANS at Re 200)
ALPHAS = (2.0, 2.6, 2.3, 2.1, 2.45)


@pytest.mark.parametrize("kind, eta_threshold, run_fom", [("euler", 1e-9, False), ("rans", 1e-7, True)])
def test_windtunnel_accepts_the_rom_between_snapshots(in_tmp_dir, kind, eta_threshold, run_fom):
    """Two FOM snapshots, a rejected ROM (third snapshot), then two accepted
    ROM solutions. Without run_fom_every_evaluation (Euler) the accepted ROM
    solution is the output and the warm start and no FOM runs; with it (RANS)
    the FOM runs too and the coefficient errors are recorded."""
    rom = ReducedOrderModel(rb_size=5, eta_threshold=eta_threshold, run_fom_every_evaluation=run_fom)
    wt = make_windtunnel(kind, rom=rom)
    out = [wt.solve_forward(np.radians(a)) for a in ALPHAS]
    assert all(c for _, c, _ in out)
    assert rom.n_snapshots == 3          # the two first evaluations and the rejected third
    if MPI.COMM_WORLD.rank == 0:
        ds = wt.data_store
        assert [acc for _, acc in ds.ROM_meets_threshold] == [False, True, True]
        assert len(ds.global_snapshot_sing_vals_per_iteration) == 3
        assert len(ds.FOM_walltime) == (len(ALPHAS) if run_fom else 3)
        assert len(ds.rom_coefficient_errors) == (3 if run_fom else 1)
        for _, dcd, dcl, dcm in ds.rom_coefficient_errors[1:]:
            assert max(abs(dcd), abs(dcl), abs(dcm)) < 1e-6
    u = out[-1][0]
    assert np.array_equal(wt.previous_solution, u)
    if not run_fom:
        # the output is the ROM solution itself
        m = wt.sim_model
        m.set_u_vec(u)
        assert m.relative_residual_norm(m._assemble_physical_residual()) < eta_threshold


def test_windtunnel_falls_back_to_the_fom(in_tmp_dir):
    """eta_threshold 0 never accepts the ROM: every evaluation is the FOM
    from the same warm start, identical to a windtunnel without a ROM (to the
    last bit serially; in parallel two plain windtunnels already differ by
    ~1e-15 run to run)."""
    rom = ReducedOrderModel(rb_size=5, eta_threshold=0.0)
    wt_rom = make_windtunnel("euler", rom=rom, suffix="fallback")
    wt_fom = make_windtunnel("euler", suffix="plain")
    for a in ALPHAS[:4]:
        u_rom, c_rom, f_rom = wt_rom.solve_forward(np.radians(a))
        u_fom, c_fom, f_fom = wt_fom.solve_forward(np.radians(a))
        assert c_rom and c_fom
        assert u_rom == pytest.approx(u_fom, rel=1e-12, abs=1e-12)
        assert f_rom == pytest.approx(f_fom, rel=1e-12, abs=1e-12)
    assert rom.n_snapshots == 4
    if MPI.COMM_WORLD.rank == 0:
        assert [acc for _, acc in wt_rom.data_store.ROM_meets_threshold] == [False, False]


# ----------------------------------------------------------------------
# LSPG and its stopping rules on a linear stand-in model
# ----------------------------------------------------------------------
class _LinearProblem:
    """R(u) = K u - f with a non-symmetric K, in the problem interface the
    ROM uses (F, J, _A, _b)."""

    def __init__(self, K, f, u):
        self._A, self._b, self.K, self.f, self.u = K, f.duplicate(), K, f, u

    def F(self, x, b):
        self.K.mult(x, b)
        b.axpy(-1.0, self.f)

    def J(self, x, A):
        pass


class _LinearModel:
    """The host interface of core/reduced_order_model.py, for R(u) = K u - f."""

    def __init__(self, comm, n=60, n_state=3, seed=11):
        import contextlib
        from types import SimpleNamespace
        rng = np.random.default_rng(seed)
        self.n_state = n_state
        x = PETSc.Vec().createMPI((PETSc.DECIDE, n), comm=comm)
        rs, re = x.getOwnershipRange()
        K = PETSc.Mat().createAIJ(size=((re - rs, n), (re - rs, n)), nnz=(3, 3), comm=comm)
        K.setUp()
        for i in range(rs, re):
            K.setValue(i, i, 3.0 + np.sin(i))
            if i + 1 < n:
                K.setValue(i, i + 1, 0.7)          # upper only: non-symmetric
        K.assemble()
        f = x.duplicate()
        f.array[:] = np.random.default_rng(seed + 1).standard_normal(n)[rs:re]
        self.u_vec = SimpleNamespace(x=SimpleNamespace(petsc_vec=x, scatter_forward=lambda: None))
        self.problem = _LinearProblem(K, f, self.u_vec)
        self.mesh_obj = SimpleNamespace(mesh=SimpleNamespace(comm=comm))
        self.steady_residual = contextlib.nullcontext
        self.rng, self.n, self.rs, self.re = rng, n, rs, re

    def set_up_solver(self):
        pass

    def positivity_step_length(self, x, dx):
        return 1.0, 1.0

    def state_scales(self):
        return [1.0] * self.n_state

    def _assemble_physical_residual(self):
        r = self.problem._b
        self.problem.F(self.u_vec.x.petsc_vec, r)
        return r.norm()

    def relative_residual_norm(self, full_norm):
        return full_norm / self.u_vec.x.petsc_vec.norm()


def _linear_rom(model, k=4, **kw):
    rom = ReducedOrderModel(rb_size=k, inner_product="euclidean", state_scaling=None, **kw)
    rom.set_up(model)
    x = model.u_vec.x.petsc_vec
    for j in range(k):
        x.array[:] = model.rng.standard_normal(model.n)[model.rs:model.re]
        rom.add_snapshot(model, np.array([float(j)]))
    x.set(0.0)
    return rom


def test_lspg_solves_the_least_squares_normal_equations():
    """The ROM solution satisfies (K V)^T (K V a - f) = 0 (LSPG), not the
    Galerkin V^T (K V a - f) = 0, which differs for a non-symmetric K."""
    comm = MPI.COMM_WORLD
    model = _LinearModel(comm)
    rom = _linear_rom(model)
    result = rom.solve(model, np.array([0.5]))
    assert result.converged
    V = np.vstack(comm.allgather(rom.pod.basis.getDenseArray().copy()))
    K = model.problem.K
    KV = np.column_stack([global_array(comm, _matvec(K, V[:, j], model)) for j in range(V.shape[1])])
    u = global_array(comm, result.solution)
    f = global_array(comm, model.problem.f.getArray())
    r = global_array(comm, _matvec(K, u, model)) - f
    assert np.linalg.norm(KV.T @ r) <= 1e-10 * np.linalg.norm(KV.T @ f)
    assert np.linalg.norm(V.T @ r) > 1e-3 * np.linalg.norm(V.T @ f)


def _matvec(K, v_global, model):
    x = model.u_vec.x.petsc_vec.duplicate()
    x.array[:] = v_global[model.rs:model.re]
    y = x.duplicate()
    K.mult(x, y)
    out = y.getArray().copy()
    x.destroy()
    y.destroy()
    return out


def test_outer_passes_and_absolute_g_limit():
    """newton_max_it 1 and g_conv_limit 0: max_outer_iterations passes of one
    iteration, never converged (benchmark_pod.py's budget structure)."""
    model = _LinearModel(MPI.COMM_WORLD)
    rom = _linear_rom(model, newton_max_it=1, max_outer_iterations=3, g_conv_limit=0.0)
    result = rom.solve(model, np.array([0.5]))
    assert result.iterations == 3
    assert not result.converged
