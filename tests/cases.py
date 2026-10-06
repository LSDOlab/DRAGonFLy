"""
Small, inline test cases shared by the test modules.

Everything here builds its mesh in-process (no mesh files) and is cheap
enough to construct many times per test session. Every builder takes the
communicator, so the same cases run serially and under mpirun.
"""
import numpy as np
import dolfinx
from mpi4py import MPI

from dragonfly_sim.core.mesh_manager import Mesh
from dragonfly_sim.core.Euler_model import CompressibleEulerModel

INLET_TAG, OUTLET_TAG, WALL_TAG, FARFIELD_TAG = 3, 4, 2, 6

FREESTREAM = {'inlet': {'rho': 1.0, 'p': 1.0, 'M': 0.5, 'alpha': np.radians(2.0)},
              'outlet': {'p': 1.0}}


def jittered_square(n=8, comm=None, jitter=0.01, seed=3):
    """Unit-square quad mesh with its interior nodes jittered (boundary kept straight)."""
    comm = comm if comm is not None else MPI.COMM_WORLD
    msh = dolfinx.mesh.create_unit_square(comm, n, n, dolfinx.mesh.CellType.quadrilateral)
    x = msh.geometry.x
    # the jitter is a smooth function of position, so every rank moves its
    # copy of a shared node identically
    interior = (x[:, 0] > 1e-12) & (x[:, 0] < 1 - 1e-12) & (x[:, 1] > 1e-12) & (x[:, 1] < 1 - 1e-12)
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(4)
    x[interior, 0] += jitter * np.sin(7.1 * x[interior, 0] + a[0]) * np.sin(5.3 * x[interior, 1] + a[1])
    x[interior, 1] += jitter * np.sin(6.7 * x[interior, 0] + a[2]) * np.sin(4.9 * x[interior, 1] + a[3])
    return msh


def channel_tags(mesh_obj):
    """Left side inlet, right side outlet, top and bottom slip walls."""
    mid = mesh_obj.bdry_midpoints
    inlet = mid[:, 0] < 1e-8
    outlet = mid[:, 0] > 1 - 1e-8
    wall = ~(inlet | outlet)
    return inlet, outlet, wall


def euler_channel_model(n=8, comm=None, quadrature_degree=1, bc_dict=FREESTREAM, build_weakform=True):
    """
    The release Euler model on a jittered unit-square channel: subsonic
    inflow on the left, subsonic outflow on the right, slip walls top and
    bottom. Mirrors DG_windtunnel_model.set_up_sim's order of calls.
    """
    msh = jittered_square(n, comm)
    mesh_obj = Mesh(msh, lambda pts: np.zeros(pts.shape[1], dtype=bool))
    inlet, outlet, wall = channel_tags(mesh_obj)
    mesh_obj.tag_boundary_facets([(inlet, INLET_TAG), (outlet, OUTLET_TAG), (wall, WALL_TAG)])
    meta = {'quadrature_degree': quadrature_degree}
    mesh_obj.form_manager.build_boundary_measure(measure_metadata=meta)
    mesh_obj.form_manager.build_cell_and_interior_facet_measures(measure_metadata=meta)

    model = CompressibleEulerModel(mesh_obj, poly_order=0)
    model.residual_conv_limit = 1e-9
    model.max_newton_iterations = 100
    model.define_inlet_outlet_conditions(bc_dict)
    model.define_elements_functionspaces()
    model.define_trial_testfunctions()
    model.compute_initial_conditions_from_inlet()
    model.interpolate_solution_vector()
    model.define_subsonic_inflow_bc(INLET_TAG)
    model.define_subsonic_outflow_bc(OUTLET_TAG)
    model.define_slipwall_bc(WALL_TAG)
    if build_weakform:
        model.compute_weakform()
    return model


def seed_mean_flow(model, amplitude=1.0):
    """
    A smooth, strictly physical mean-flow state (no vacuum, no negative
    pressure) written into model.u_vec; extra components are left as they are.
    """
    V = model.functionspaces["V"]
    coords = V.tabulate_dof_coordinates()
    bs = V.dofmap.index_map_bs
    x, y = coords[:, 0], coords[:, 1]
    rho = 1.0 + amplitude * 0.3 * np.sin(3.0 * x) * np.cos(2.0 * y)
    ux = 0.5 + amplitude * 0.2 * np.cos(2.0 * x + y)
    uy = -0.05 + amplitude * 0.1 * np.sin(x - 3.0 * y)
    p = 1.0 + amplitude * 0.25 * np.cos(x + 2.0 * y)
    arr = model.u_vec.x.array.reshape(-1, bs).copy()
    gamma = model.gamma
    arr[:, 0] = rho
    arr[:, 1] = rho * ux
    arr[:, 2] = rho * uy
    arr[:, 3] = p / (gamma - 1.0) + 0.5 * rho * (ux ** 2 + uy ** 2)
    model.u_vec.x.array[:] = arr.ravel()
    model.u_vec.x.scatter_forward()


def owned_cell_order(V):
    """Original (input-mesh) cell index of every owned DG0 cell, for rank-independent comparisons."""
    msh = V.mesh
    n_own = msh.topology.index_map(msh.topology.dim).size_local
    return np.asarray(msh.topology.original_cell_index[:n_own], dtype=np.int64)


def gather_by_original_cell(V, local_array, comm=None):
    """Gather an owned per-cell array (n_own, k) onto every rank, ordered by original cell index."""
    comm = comm if comm is not None else V.mesh.comm
    order = owned_cell_order(V)
    pairs = comm.allgather((order, np.asarray(local_array)[:order.size]))
    idx = np.concatenate([p[0] for p in pairs])
    val = np.concatenate([p[1] for p in pairs])
    return val[np.argsort(idx)]


# ----------------------------------------------------------------------
# RANS
# ----------------------------------------------------------------------
def rans_square_model(turbulence_model, n=6, comm=None, jitter=0.01, Re=50.0, quadrature_degree=2,
                      wall_bottom=False, **rans_kwargs):
    """
    CompressibleRANSModel on the unit square, function spaces defined,
    no boundary conditions yet. Every boundary facet carries tag 1, or with
    wall_bottom the bottom side carries WALL_TAG.
    """
    from dragonfly_sim.core.RANS_model import CompressibleRANSModel
    comm = comm if comm is not None else MPI.COMM_WORLD
    msh = dolfinx.mesh.create_unit_square(comm, n, n, dolfinx.mesh.CellType.quadrilateral)
    if jitter:
        x = msh.geometry.x
        interior = (x[:, 0] > 1e-12) & (x[:, 0] < 1 - 1e-12) & (x[:, 1] > 1e-12) & (x[:, 1] < 1 - 1e-12)
        x[interior, 0] += jitter / n * np.sin(17.0 * x[interior, 0] + 5.0 * x[interior, 1])
        x[interior, 1] += jitter / n * np.cos(11.0 * x[interior, 0] - 7.0 * x[interior, 1])
    mesh_obj = Mesh(msh, lambda pts: np.zeros(pts.shape[1], dtype=bool))
    if wall_bottom:
        bottom = mesh_obj.bdry_midpoints[:, 1] < 1e-10
        mesh_obj.tag_boundary_facets([(~bottom, 1), (bottom, WALL_TAG)])
    else:
        mesh_obj.tag_boundary_facets([(np.arange(mesh_obj.bdry_facets.size, dtype=np.int32), 1)])
    meta = {"quadrature_degree": quadrature_degree}
    mesh_obj.form_manager.build_boundary_measure(measure_metadata=meta)
    mesh_obj.form_manager.build_cell_and_interior_facet_measures(measure_metadata=meta)
    model = CompressibleRANSModel(mesh_obj, Re=Re, turbulence_model=turbulence_model, **rans_kwargs)
    model.define_inlet_outlet_conditions(
        {'inlet': {'rho': 1.0, 'M': 0.5, 'p': 1.0, 'alpha': 0.0}, 'outlet': {'p': 1.0}})
    model.define_elements_functionspaces()
    model.define_trial_testfunctions()
    return model


def manufactured_state(model):
    """A smooth, physical manufactured state (with a positive nu_tilde)."""
    import ufl
    gamma = model.gamma
    x = ufl.SpatialCoordinate(model.mesh_obj.mesh)
    pi = np.pi
    rho = 1.0 + 0.2 * ufl.sin(pi * x[0]) * ufl.cos(0.5 * pi * x[1])
    u = 0.5 + 0.15 * ufl.cos(pi * x[0] + 0.3) * ufl.sin(pi * x[1])
    v = -0.1 + 0.1 * ufl.sin(pi * (x[0] - x[1]))
    p = 1.0 + 0.15 * ufl.cos(pi * x[0]) * ufl.sin(0.5 * pi * x[1] + 0.2)
    comps = [rho, rho * u, rho * v, p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)]
    if model.n_extra:
        nu_t = model.mu_inf_const * (3.0 + 2.0 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]))
        comps.append(rho * nu_t)
    return ufl.as_vector(comps), x


def mms_rans_model(turbulence_model, n, Re=2.0, jitter=0.01, **rans_kwargs):
    """
    A RANS model whose exact discrete-continuous target is the manufactured
    state U_e: the strong-form residual of U_e is subtracted as a source,
    and U_e is imposed weakly on the whole boundary through the model's own
    inviscid and viscous boundary terms. Returns (model, U_e).
    """
    import ufl
    from dragonfly_sim.utils.NS_utils import extended_flux
    model = rans_square_model(turbulence_model, n=n, Re=Re, jitter=jitter, **rans_kwargs)
    U_e, x = manufactured_state(model)
    nm, gamma = model.n_mean, model.gamma
    if model.n_extra:
        # a smooth stand-in for the wall distance: the source only needs to
        # be the same function in the discrete and the exact problem
        model.wall_distance = 0.2 + x[1]
    strong = ufl.div(extended_flux(U_e, gamma, nm)) - ufl.div(model.viscous_flux(U_e, ufl.grad(U_e)))
    if model.n_extra:
        mu = model.laminar_viscosity(U_e)
        S = model.turbulence_model.sources(U_e, ufl.grad(U_e), mu, model.wall_distance, nm, gamma)
        strong = strong - ufl.as_vector([0.0] * nm + S)
    model.flow.residual_terms.append(lambda: - ufl.inner(strong, model.v_vec) * model.forms.dx)
    model.add_boundary_condition(1, U_e, "dirichlet")
    model.compute_weakform()
    return model, U_e


def l2_error(model, U_e):
    import ufl
    e = model.u_vec - U_e
    comm = model.mesh_obj.mesh.comm
    out = []
    for k in range(model.n_state):
        val = dolfinx.fem.assemble_scalar(dolfinx.fem.form(e[k]**2 * model.forms.dx))
        ref = dolfinx.fem.assemble_scalar(dolfinx.fem.form(U_e[k]**2 * model.forms.dx))
        out.append(np.sqrt(comm.allreduce(val, MPI.SUM) / comm.allreduce(ref, MPI.SUM)))
    return np.array(out)
