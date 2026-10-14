"""
The four speed-ups of the adjoint of a model with a condensed Jacobian (see
DG_windtunnel_model.set_adjoint_options) are exact reformulations: each must give
the same operators and the same derivatives as the plain path.

  1. the cell-centre matrix dc/dx built vectorized and once,
  2. the Green-Gauss Jacobians (and the dR/dx operator, dR/dalpha) cached per state,
  3. dG/dU applied as W (x) I plus a boundary part instead of an assembled matrix,
  4. the transposed condensed Jacobian applied without forming it.

Checked on a small RANS flow in 2D (quads) and 3D (hexes) with a no-slip wall, a far
field and, in 3D, a symmetry plane.
"""
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import csdl_alpha as csdl

from dragonfly_sim.core.RANS_model import CompressibleRANSModel

from test_moments import in_tmp_dir, make_windtunnel, make_windtunnel_3d, BC  # noqa: F401 (fixture)

MOMENTS_2D = [("pitch", (0.25, 0.), "pitch")]
MOMENTS_3D = [("pitch", (0.25, 0., 0.), "pitch"), ("root_bending", (0.25, 0., 0.), (1., 0., 0.))]
RANS = dict(model_class=CompressibleRANSModel, model_kwargs={'Re': 1e3})


def solved_windtunnel(dim):
    wt = make_windtunnel(moments=MOMENTS_2D, **RANS) if dim == 2 else make_windtunnel_3d(moments=MOMENTS_3D, **RANS)
    wt.max_newton_iterations = 400
    wt.linear_solver_atol, wt.linear_residual_accept_limit = 1e-9, 1e-8
    u, converged, _ = wt.solve_forward()
    assert converged
    wt.sim_model.set_u_vec(u)
    wt.sim_model.update_derived_fields()
    return wt, u


def mat_difference(A, B):
    """|A - B|_F / |A|_F for two assembled matrices."""
    C = A.duplicate(copy=True)
    C.axpy(-1.0, B, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
    return C.norm(PETSc.NormType.FROBENIUS) / A.norm(PETSc.NormType.FROBENIUS)


def random_vec(M, left, seed):
    v = M.createVecLeft() if left else M.createVecRight()
    n = v.getLocalSize()
    rng = np.random.default_rng(seed + 1000 * MPI.COMM_WORLD.rank)
    v.setArray(rng.standard_normal(n))
    return v


def rel(a, b):
    return (a - b).norm() / a.norm()


@pytest.mark.parametrize("dim", [2, 3])
def test_vectorized_cell_centre_matrix_equals_the_loop(dim, in_tmp_dir):
    wt, _ = solved_windtunnel(dim)
    rec = wt.sim_model.reconstruction
    centres = rec.centres
    loop = centres._mesh_jacobian_loop()
    fast = centres._mesh_jacobian_vectorized()
    assert mat_difference(loop, fast) < 1e-14
    # cached: the same object every call, only with the option on
    rec.fast_cell_centre_jacobian = True
    assert centres.mesh_jacobian() is centres.mesh_jacobian()
    centres.discard(centres.mesh_jacobian())            # the cache keeps it
    assert mat_difference(loop, centres.mesh_jacobian()) < 1e-14
    rec.fast_cell_centre_jacobian = False
    other = centres.mesh_jacobian()
    assert other is not centres._mesh_jac
    other.destroy()


@pytest.mark.parametrize("dim", [2, 3])
def test_structured_gradient_operator_equals_the_assembled_matrix(dim, in_tmp_dir):
    wt, u = solved_windtunnel(dim)
    rec, grad = wt.sim_model.reconstruction, wt.sim_model.reconstruction.gradient
    rec.structured_state_jacobian = False
    B = grad.state_jacobian()
    rec.structured_state_jacobian = True
    S = grad.state_jacobian()
    for seed in (1, 2):
        x, v = random_vec(B, False, seed), random_vec(B, True, seed)
        y_ref, y = B.createVecLeft(), S.createVecLeft()
        B.mult(x, y_ref); S.mult(x, y)
        assert rel(y_ref, y) < 1e-12
        z_ref, z = B.createVecRight(), S.createVecRight()
        B.multTranspose(v, z_ref); S.multTranspose(v, z)
        assert rel(z_ref, z) < 1e-12
        # the way the chain rule uses it: g += B^T v
        g_ref, g = z_ref.duplicate(), z_ref.duplicate()
        g_ref.set(1.0); g.set(1.0)
        B.multTransposeAdd(v, g_ref, g_ref); S.multTransposeAdd(v, g, g)
        assert rel(g_ref, g) < 1e-12
    grad.discard(S)
    B.destroy()


@pytest.mark.parametrize("dim", [2, 3])
def test_matrix_free_adjoint_operator_equals_the_formed_one(dim, in_tmp_dir):
    wt, u = solved_windtunnel(dim)
    model, rec = wt.sim_model, wt.sim_model.reconstruction
    kw = dict(transpose=True, prefix="t_", rtol=1e-12, atol=1e-12, max_it=500, release_operands=True)
    ref = model.linearized_solver(**kw)
    A_ref, P_ref = ref.getOperators()
    for structured in (False, True):
        rec.structured_state_jacobian = structured
        free = model.linearized_solver(matrix_free=True, **kw)
        A, P = free.getOperators()
        assert A.getType() == PETSc.Mat.Type.PYTHON
        assert mat_difference(P_ref, P) < 1e-14                 # the same preconditioning matrix
        for seed in (3, 4):
            x = random_vec(A_ref, False, seed)
            y_ref, y = A_ref.createVecLeft(), A.createVecLeft()
            A_ref.mult(x, y_ref); A.mult(x, y)
            assert rel(y_ref, y) < 1e-12, (dim, structured)
        # and it solves: the same adjoint solution
        b = random_vec(A_ref, True, 9)
        s_ref, s = A_ref.createVecRight(), A.createVecRight()
        ref.solve(b, s_ref); free.solve(b, s)
        assert rel(s_ref, s) < 1e-8
        model.destroy_adjoint_operator(free)
    model.destroy_adjoint_operator(ref)
    rec.structured_state_jacobian = False


def graph(wt):
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    alpha = csdl.Variable(name='alpha', value=np.array([BC['inlet']['alpha']]))
    alpha.set_as_design_variable()
    u_vec = wt.evaluate(alpha=alpha)
    out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
    outputs = [out.D, out.L] + [getattr(out, "M_" + m.name) for m in wt.postprocessor.moment_specs]
    for i, o in enumerate(outputs):
        o.add_name("out%d" % i)
    outputs[0].set_as_objective()
    for o in outputs[1:]:
        o.set_as_constraint(lower=-1e6)
    recorder.stop()
    return csdl.experimental.JaxSimulator(recorder, gpu=False), alpha, outputs


OPTIONS = dict(cell_centre_matrix=True, cache_jacobians=True, structured_gradient=True,
               matrix_free=True, cache_jacvec_operators=True)


@pytest.mark.parametrize("dim", [2, 3])
def test_all_speedups_give_the_same_totals(dim, in_tmp_dir):
    """The totals of every output, at two angles of attack in a row (so the caches are
    invalidated and rebuilt), with all options off and all on; also with one option on
    at a time."""
    wt = (make_windtunnel(moments=MOMENTS_2D, **RANS) if dim == 2
          else make_windtunnel_3d(moments=MOMENTS_3D, **RANS))
    wt.max_newton_iterations = 400
    wt.linear_solver_atol, wt.linear_residual_accept_limit = 1e-9, 1e-8
    sim, alpha, outputs = graph(wt)
    alphas = np.radians([2.0, 2.05])

    def totals():
        result = []
        for a in alphas:
            sim[alpha] = np.array([a])
            sim.run()
            t = sim.compute_totals()
            result.append([float(np.asarray(t[o, alpha]).ravel()[0]) for o in outputs])
        return np.array(result)

    wt.set_adjoint_options()
    base = totals()
    configs = [dict(OPTIONS)] + [{k: True} for k in OPTIONS]
    for cfg in configs:
        wt.set_adjoint_options(**cfg)
        got = totals()
        np.testing.assert_allclose(got, base, rtol=2e-6, atol=1e-12, err_msg=str(cfg))
    # the caches hold the last design until the next forward solve, then are freed
    wt.set_adjoint_options(**OPTIONS)
    totals()
    assert wt._adjoint_cache is not None and wt.sim_model.reconstruction.gradient._cache is not None
    wt.solve_forward(alphas[0])
    assert wt._adjoint_cache is None and wt.sim_model.reconstruction.gradient._cache is None


def test_speedups_with_mesh_motion_give_the_same_totals(in_tmp_dir):
    """2D RANS with an FFD shape parameterization: the totals with respect to the shape
    parameters and alpha, which also go through dR/dx, dG/dx and the cell-centre
    matrix, with all options off and on, at two designs."""
    from dragonfly_sim.core.shape_design import FFDShapeParameterization, ControlPointMotions
    from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
    from cases import jittered_square
    from test_windtunnel_forward import bottom_wall

    recorder = csdl.Recorder(inline=False)
    recorder.start()
    shape = FFDShapeParameterization([5, 3], [2, 2], layers=[ControlPointMotions({1: 0.01})])
    # the quadrilaterals of the generated square have a negative Jacobian determinant, which a
    # mesh motion rejects as inverted cells: mirror the mesh in x
    msh = jittered_square(8)
    msh.geometry.x[:, 0] = 1.0 - msh.geometry.x[:, 0]
    wt = DG_windtunnel_model(msh, BC, shape, moments=MOMENTS_2D,
                             mesh_inner_bdry_function=bottom_wall, poly_order=0,
                             filename_suffix="test_speedups_mesh", **RANS)
    wt.write_mesh_deformation = False
    wt.max_newton_iterations = 400
    wt.linear_solver_atol, wt.linear_residual_accept_limit = 1e-9, 1e-8
    from test_moments import _WINDTUNNELS
    _WINDTUNNELS.append(wt)
    wt.set_up_sim()
    params = shape.declare_design_variables()
    alpha = csdl.Variable(name='alpha', value=np.array([BC['inlet']['alpha']]))
    alpha.set_as_design_variable()
    motion = wt.deform_mesh()
    u_vec = wt.evaluate(motion, params, alpha)
    out = wt.postprocessor.evaluate(u_vec, motion, alpha, params)
    outputs = [out.D, out.L, out.M_pitch]
    for i, o in enumerate(outputs):
        o.add_name("out%d" % i)
    outputs[0].set_as_objective()
    for o in outputs[1:]:
        o.set_as_constraint(lower=-1e6)
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    rng = np.random.default_rng(4)
    p0 = np.array(params.value, copy=True)
    designs = [p0, p0 + 2e-3 * rng.uniform(-1, 1, p0.shape)]

    def totals():
        result = []
        for p in designs:
            sim[params] = p
            sim.run()
            t = sim.compute_totals()
            result.append(np.concatenate([np.concatenate(
                [np.asarray(t[o, params]).ravel(), np.asarray(t[o, alpha]).ravel()]) for o in outputs]))
        return np.array(result)

    wt.set_adjoint_options()
    base = totals()
    assert np.abs(base).max() > 1e-6
    for cfg in [dict(OPTIONS)] + [{k: True} for k in OPTIONS]:
        wt.set_adjoint_options(**cfg)
        np.testing.assert_allclose(totals(), base, rtol=2e-5, atol=1e-9, err_msg=str(cfg))
    wt.set_adjoint_options(**OPTIONS)
    totals()
    assert wt._dRdx_cache is not None and wt._dRdalpha_cache is not None
    wt.solve_forward()
    assert wt._dRdx_cache is None and wt._adjoint_cache is None
