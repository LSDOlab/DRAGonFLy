"""
Moments of the wall force in DG_postprocessor: each (name, point, direction)
tuple gives M = d . (r x F), with its value, state, mesh and alpha derivatives.

The flow is the jittered unit square of test_windtunnel_forward.py, whose
bottom side is the wall. Derivatives are checked against central finite
differences, and the values against independent hand-written forms and
identities of the moment.
"""
import os

import numpy as np
import pytest
import ufl
import dolfinx
from mpi4py import MPI

import csdl_alpha as csdl

from dragonfly_sim.core.postprocessor import DG_postprocessor
from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.utils.postprocessor_utils import moment_about

from cases import jittered_square
from test_windtunnel_forward import BC, bottom_wall

MOMENTS = [("pitch", (0.25, 0.), "pitch"),
           ("off_axis", (0.7, 0.3), (0., 0., 1.)),
           ("scaled", (0.1, -0.2), (0., 0., -2.), 0.5)]


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
    """The windtunnel writes output files into the working directory: use one
    temporary directory, shared by all ranks."""
    comm = MPI.COMM_WORLD
    path = comm.bcast(str(tmp_path) if comm.rank == 0 else None, root=0)
    cwd = os.getcwd()
    os.chdir(path)
    yield
    while _WINDTUNNELS:
        _close_writers(_WINDTUNNELS.pop())
    comm.Barrier()
    os.chdir(cwd)


def make_windtunnel(moments=MOMENTS, **kw):
    wt = DG_windtunnel_model(jittered_square(8), BC, moments=moments, mesh_inner_bdry_function=bottom_wall,
                             poly_order=0, filename_suffix="test_moments", **kw)
    wt.euler_residual_conv_limit = 1e-12
    wt.set_up_sim()
    _WINDTUNNELS.append(wt)
    return wt


def spec_of(post, name):
    return next(s for s in post.moment_specs if s.name == name)


def global_sum(post, form):
    return post.mesh_obj.mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(form)), MPI.SUM)


def test_pitch_preset_is_the_nose_up_moment(in_tmp_dir):
    wt = make_windtunnel()
    u, converged, _ = wt.solve_forward()
    assert converged
    post = wt.postprocessor
    # an independent hand-written form of the old pitching moment, F_x r_y - F_y r_x
    force = post._pressure_force_vector(u)
    r = ufl.SpatialCoordinate(post.mesh_obj.mesh) - ufl.as_vector([0.25, 0.])
    hand = global_sum(post, (force[0]*r[1] - force[1]*r[0]) * post.mesh_obj.form_manager.ds(post.wall_meshtag_val))
    M, c_m = post.compute_moment(u, spec_of(post, "pitch"))
    assert abs(M - hand) < 1e-12 * abs(hand)
    assert abs(M) > 1e-4                      # the comparison is not between two zeros
    # the coefficient: M / (q S_ref c_ref), the reference chord being 1 in 2D
    q = global_sum(post, post.C_infty * ufl.dx(domain=post.mesh_obj.mesh))     # the unit square has area 1
    assert abs(c_m - M / q) < 1e-12 * abs(c_m)


def test_direction_is_normalised_and_signed(in_tmp_dir):
    wt = make_windtunnel([("a", (0.3, 0.1), (0., 0., 1.)), ("b", (0.3, 0.1), (0., 0., 7.)),
                          ("c", (0.3, 0.1), (0., 0., -1.)), ("d", (0.3, 0.1), -1.)])
    u, _, _ = wt.solve_forward()
    post = wt.postprocessor
    Ma, Mb, Mc, Md = (post.compute_moment(u, spec_of(post, n))[0] for n in "abcd")
    assert Mb == pytest.approx(Ma, rel=1e-14) and Mc == pytest.approx(-Ma, rel=1e-14)
    assert Md == Mc                           # the scalar shorthand of the out-of-plane axis


def test_reference_length_only_scales_the_coefficient(in_tmp_dir):
    wt = make_windtunnel()
    u, _, forces = wt.solve_forward()
    post = wt.postprocessor
    M_a, c_a = post.compute_moment(u, spec_of(post, "off_axis"))
    # "scaled" has direction -z, twice as long, and ref_length 0.5
    M_b, c_b = post.compute_moment(u, spec_of(post, "scaled"))
    assert forces["moments"]["scaled"] == dict(M=M_b, c_m=c_b)
    assert abs(c_b / M_b) == pytest.approx(abs(c_a / M_a) / 0.5, rel=1e-12)


def test_moment_about_a_moved_point(in_tmp_dir):
    """M(x1) = M(x0) - F_tot . (d x (x1 - x0)): the same force, another lever arm."""
    wt = make_windtunnel([("m0", (0.25, 0.), (0., 0., 1.)), ("m1", (0.8, -0.35), (0., 0., 1.))])
    u, _, _ = wt.solve_forward()
    post = wt.postprocessor
    force = post._pressure_force_vector(u)
    ds = post.mesh_obj.form_manager.ds(post.wall_meshtag_val)
    F_tot = np.array([global_sum(post, force[i]*ds) for i in range(2)] + [0.])
    M0 = post.compute_moment(u, spec_of(post, "m0"))[0]
    M1 = post.compute_moment(u, spec_of(post, "m1"))[0]
    delta = np.array([0.8 - 0.25, -0.35, 0.])
    assert M1 == pytest.approx(M0 - F_tot @ np.cross([0., 0., 1.], delta), rel=1e-10)


@pytest.mark.parametrize("direction", [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (0.3, -1.2, 0.8)])
def test_moment_about_in_3D(direction):
    """moment_about is d . (r x F) for any direction, on constants over the unit cube."""
    comm = MPI.COMM_WORLD
    msh = dolfinx.mesh.create_unit_cube(comm, 1, 1, 1)
    rng = np.random.default_rng(2)
    F, r = rng.standard_normal(3), rng.standard_normal(3)
    d = np.array(direction) / np.linalg.norm(direction)
    value = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        moment_about(dolfinx.fem.Constant(msh, F), dolfinx.fem.Constant(msh, r),
                     dolfinx.fem.Constant(msh, d), 3) * ufl.dx))
    assert comm.allreduce(value, MPI.SUM) == pytest.approx(d @ np.cross(r, F), rel=1e-13)


def check_state_and_mesh_derivatives(wt, u):
    post, mo = wt.postprocessor, wt.mesh
    comm = mo.mesh.comm
    gdim = mo.mesh.geometry.dim
    rng = np.random.default_rng(11)
    du = rng.standard_normal(u.size) * np.abs(u)
    X = mo.mesh.geometry.x[:mo.n_owned_nodes, :gdim]
    dX = np.stack([np.sin(3*X[:, i] + X[:, (i + 1) % gdim]) * (X[:, (i + 1) % gdim] + 0.2)
                   for i in range(gdim)], axis=1)
    eps = 1e-6
    for spec in post.moment_specs:
        g = post.compute_moment_deriv(u, spec)[:u.size]
        fd = (post.compute_moment(u + eps*du, spec)[0] - post.compute_moment(u - eps*du, spec)[0]) / (2*eps)
        an = comm.allreduce(float(g @ du), MPI.SUM)
        assert abs(fd - an) < 1e-7 * abs(an), (spec.name, fd, an)

        gx = post.compute_moment_meshderiv(u, spec)
        mo.apply_node_motions(eps*dX)
        Mp = post.compute_moment(u, spec)[0]
        mo.apply_node_motions(-eps*dX)
        Mm = post.compute_moment(u, spec)[0]
        mo.reset_nodes()
        an = comm.allreduce(float(gx[:dX.size] @ dX.ravel()), MPI.SUM)
        assert abs((Mp - Mm) / (2*eps) - an) < 1e-6 * abs(an), (spec.name, (Mp - Mm) / (2*eps), an)


def test_state_and_mesh_derivatives_match_finite_differences(in_tmp_dir):
    wt = make_windtunnel()
    u, _, _ = wt.solve_forward()
    check_state_and_mesh_derivatives(wt, u)


# ----------------------------------------------------------------------
# 3D: a unit cube whose bottom face is the wall and y = 0 the symmetry plane
# ----------------------------------------------------------------------
MOMENTS_3D = [("pitch", (0.25, 0., 0.), "pitch"),
              ("root_bending", (0.25, 0., 0.), (1., 0., 0.)),
              ("oblique", (0.4, 0.1, 0.3), (0.3, -1.2, 0.8), 2.0)]


def jittered_cube(n=4, jitter=0.02):
    msh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, dolfinx.mesh.CellType.hexahedron)
    x = msh.geometry.x
    inner = np.all((x > 1e-12) & (x < 1 - 1e-12), axis=1)
    x[inner, 0] += jitter*np.sin(7.1*x[inner, 1])*np.sin(5.3*x[inner, 2])
    x[inner, 1] += jitter*np.sin(6.7*x[inner, 2])*np.sin(4.9*x[inner, 0])
    x[inner, 2] += jitter*np.sin(5.9*x[inner, 0])*np.sin(6.1*x[inner, 1])
    return msh


def make_windtunnel_3d(moments=MOMENTS_3D, n=4, **kw):
    wt = DG_windtunnel_model(jittered_cube(n), BC, moments=moments, mesh_inner_bdry_function=lambda x: x[2] < 1e-8,
                             poly_order=0, filename_suffix="test_moments_3d", **kw)
    wt.euler_residual_conv_limit = 1e-12
    wt.set_up_sim()
    _WINDTUNNELS.append(wt)
    return wt


def test_moments_in_3D(in_tmp_dir):
    wt = make_windtunnel_3d()
    u, converged, _ = wt.solve_forward()
    assert converged
    post = wt.postprocessor
    force = post._pressure_force_vector(u)
    ds = post.mesh_obj.form_manager.ds(post.wall_meshtag_val)
    r = ufl.SpatialCoordinate(post.mesh_obj.mesh) - ufl.as_vector([0.25, 0., 0.])
    # independent forms: (r x F)_y for the pitch preset, (r x F)_x for the root bending moment
    M_pitch = global_sum(post, (r[2]*force[0] - r[0]*force[2]) * ds)
    M_bend = global_sum(post, (r[1]*force[2] - r[2]*force[1]) * ds)
    assert post.compute_moment(u, spec_of(post, "pitch"))[0] == pytest.approx(M_pitch, rel=1e-12)
    assert post.compute_moment(u, spec_of(post, "root_bending"))[0] == pytest.approx(M_bend, rel=1e-12)
    assert abs(M_bend) > 1e-3
    # the moment about another point, along the normalised oblique direction
    F_tot = np.array([global_sum(post, force[i]*ds) for i in range(3)])
    x0, d = np.array([0.4, 0.1, 0.3]), np.array([0.3, -1.2, 0.8])
    d = d / np.linalg.norm(d)
    ref = d @ (np.array([global_sum(post, (r[1]*force[2] - r[2]*force[1])*ds),
                         global_sum(post, (r[2]*force[0] - r[0]*force[2])*ds),
                         global_sum(post, (r[0]*force[1] - r[1]*force[0])*ds)])
               - np.cross(x0 - np.array([0.25, 0., 0.]), F_tot))
    assert post.compute_moment(u, spec_of(post, "oblique"))[0] == pytest.approx(ref, rel=1e-10)


def test_3D_state_and_mesh_derivatives_match_finite_differences(in_tmp_dir):
    wt = make_windtunnel_3d()
    u, _, _ = wt.solve_forward()
    check_state_and_mesh_derivatives(wt, u)


def test_3D_csdl_alpha_derivative(in_tmp_dir):
    wt = make_windtunnel_3d()
    names = [spec[0] for spec in MOMENTS_3D]
    totals = alpha_totals(wt, names)
    a0, h = BC['inlet']['alpha'], 1e-5
    Mp = wt.solve_forward(a0 + h)[2]["moments"]
    Mm = wt.solve_forward(a0 - h)[2]["moments"]
    for n in names:
        fd = (Mp[n]["M"] - Mm[n]["M"]) / (2*h)
        assert abs(fd - totals[n][1]) < 1e-5 * abs(fd), (n, fd, totals[n][1])


def alpha_totals(wt, names):
    """CSDL graph of the forward solve and the moments: the totals dM/dalpha."""
    recorder = csdl.Recorder(inline=False)
    recorder.start()
    alpha = csdl.Variable(name='alpha', value=np.array([BC['inlet']['alpha']]))
    alpha.set_as_design_variable()
    u_vec = wt.evaluate(alpha=alpha)
    out = wt.postprocessor.evaluate(u_vec, alpha=alpha)
    for n in names:
        getattr(out, "M_" + n).add_name("M_" + n)
    getattr(out, "M_" + names[0]).set_as_objective()
    for n in names[1:]:
        getattr(out, "M_" + n).set_as_constraint(lower=-1e6)
    recorder.stop()
    sim = csdl.experimental.JaxSimulator(recorder, gpu=False)
    sim.run()
    totals = sim.compute_totals()
    return {n: (float(sim[getattr(out, "M_" + n)][0]),
                float(np.asarray(totals[getattr(out, "M_" + n), alpha]).ravel()[0])) for n in names}


@pytest.mark.parametrize("kind", ["euler", "rans"])
def test_csdl_alpha_derivative_of_several_moments(kind, in_tmp_dir):
    """The adjoint dM_k/dalpha of every moment, also through the RANS derived fields,
    against central differences of forward solves."""
    kw = {}
    if kind == "rans":
        from dragonfly_sim.core.RANS_model import CompressibleRANSModel
        kw = dict(model_class=CompressibleRANSModel, model_kwargs={'Re': 1e3})
    wt = make_windtunnel(**kw)
    if kind == "rans":
        wt.max_newton_iterations = 400
    names = [spec[0] for spec in MOMENTS]
    totals = alpha_totals(wt, names)
    # h = 1e-5: the pitching moment is small, and at 1e-6 the differences drown in the flow solver's tolerance
    a0, h = BC['inlet']['alpha'], 1e-5
    Mp = wt.solve_forward(a0 + h)[2]["moments"]
    Mm = wt.solve_forward(a0 - h)[2]["moments"]
    M0 = wt.solve_forward(a0)[2]["moments"]
    for n in names:
        value, derivative = totals[n]
        assert abs(M0[n]["M"] - value) < 1e-9 * abs(value), (n, M0[n]["M"], value)
        fd = (Mp[n]["M"] - Mm[n]["M"]) / (2*h)
        assert abs(fd - derivative) < 1e-5 * abs(fd), (n, fd, derivative)


INVALID_MOMENTS = [
    ([("a", (0., 0.), "pitch"), ("a", (0., 0.), "pitch")], "duplicate"),
    ([("1a", (0., 0.), "pitch")], "identifier"),
    ([("a b", (0., 0.), "pitch")], "identifier"),
    ([("a", (0., 0., 0.), "pitch")], "point needs 2"),
    ([("a", (0., 0.), (0., 0., 0.))], "zero"),
    ([("a", (0., 0.), (1., 0., 0.))], "out-of-plane"),
    ([("a", (0., 0.), (0., 0.))], "3 entries"),
    ([("a", (0., 0.), "yaw")], "preset"),
    ([("a", (0., 0.), "pitch", -1.)], "ref_length"),
    ([("a", (0., 0.))], "tuple"),
    (["pitch"], "tuple"),
]


def test_invalid_moments_are_rejected(in_tmp_dir):
    wt = make_windtunnel(moments=None)
    for moments, message in INVALID_MOMENTS:
        with pytest.raises(ValueError, match=message):
            DG_postprocessor(wt.mesh, wt.sim_model, wt.WALL_TAG, moments=moments)


def test_no_moments_is_allowed(in_tmp_dir):
    wt = make_windtunnel(moments=None)
    _, converged, forces = wt.solve_forward()
    assert converged and forces["moments"] == {}
