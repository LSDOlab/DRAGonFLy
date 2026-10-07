"""
Steady and unsteady flow analysis (no optimization): Euler or RANS at p = 0.

Steady (default): pseudo-transient continuation from the free stream.
Unsteady (--unsteady): BDF2 (BDF1 start) with a time-step ramp, Newton per
step, and step halving on failure; optionally dual time (--dual-time).

    # steady SA RANS, NACA0012 C-grid, M 0.7, alpha 2
    OMP_NUM_THREADS=1 mpirun -n 4 python flow_analysis.py --mesh <xdmf> --model sa \\
        --mach 0.7 --alpha 2 --Re 6e6

    # unsteady RANS from an impulsive start, characteristic far field
    ... --unsteady --steps 2000 --dt 0.05 --farfield riemann

    # unsteady Euler with the transverse non-reflecting far field
    ... --model euler --unsteady --farfield riemann2 --ff-relax-length 50

Every run writes into its own new folder under --out-dir: result.csv
(steady) or history.csv (unsteady), wall_*.csv (2D: per wall facet, both
ends and the middle, s, x, y, C_p, c_f), optional VTX fields, and
checkpoint_final.npz (restart with --restart).

The model is built directly from the package's classes (build_model below):
mesh tagging, measures, function spaces, boundary conditions and the weak
form, in the order DG_windtunnel_model.set_up_sim uses, but with no FFD block,
mesh warping or CSDL graph.
"""
import argparse
import os
import tempfile
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx

from dragonfly_sim.core.mesh_manager import Mesh
from dragonfly_sim.core.Euler_model import CompressibleEulerModel
from dragonfly_sim.core.RANS_model import CompressibleRANSModel
from dragonfly_sim.core.turbulence_models import Laminar, SpalartAllmarasNeg
from dragonfly_sim.core.farfield import TransverseFarfield
from dragonfly_sim.utils.mesh_manager_utils import airfoil_inner_bdry_function, wing_inner_bdry_function
from dragonfly_sim.utils.mesh_io_utils import load_dolfinx_mesh
from dragonfly_sim.utils.postprocessor_utils import moment_integrand
from dragonfly_sim.core.time_integration import StepSizeController, CFLController
from dragonfly_sim.utils.checkpoint import save_checkpoint, load_checkpoint
from dragonfly_sim.utils.Euler_utils import pressure
from dragonfly_sim.utils.filewriter import FileWriter

RESTART_KEYS = ("mesh", "mach", "alpha", "Re", "model", "farfield")


# ----------------------------------------------------------------------
# Model set-up and force coefficients. As DG_windtunnel_model.set_up_sim
# does it, without the mesh warper, the FFD block and CSDL: a forward
# analysis on a fixed mesh.
# ----------------------------------------------------------------------
WALL_TAG, INLET_TAG, OUTLET_TAG, SYMMETRY_TAG, FARFIELD_TAG = 2, 3, 4, 5, 6

GEOMETRIES = {
    # mesh_name: the XDMF grid name; aero_centre: moment reference point
    "airfoil": dict(inner=airfoil_inner_bdry_function, aero_centre=(0.25, 0.0), mesh_name="mesh"),
    "wing": dict(inner=wing_inner_bdry_function, aero_centre=(0.25, 0.0, 0.0), mesh_name="Grid"),
}


def build_model(dolfinx_mesh, model="sa", mach=0.5, alpha_deg=0.0, Re=6e6, geometry="airfoil",
                farfield="split", gamma=1.4, T_inf=300.0, quadrature_degree=1,
                asm_overlap=None, ilu_levels=None, direct=False, muscl=True, limiter=None,
                limiter_eps=1e-2, jacobian_mode="compact_pc", ff_beta=0.5, ff_relax_length=50.0,
                inner_bdry_function=None):
    """
    Build a ready-to-solve model on `dolfinx_mesh`.

    model     "euler", "laminar" (Navier-Stokes) or "sa" (RANS, SA-neg)
    farfield  "split"    subsonic inflow/outflow split perpendicular to the
                         free stream (DG_windtunnel_model's default)
              "riemann"  characteristic far field on the whole outer boundary
              "riemann2" riemann with the transverse correction (2D,
                         unsteady only; see core/farfield.py)

    The body (inner boundary) is the model's wall: slip for Euler, adiabatic
    no-slip otherwise. In 3D the y = 0 plane is a symmetry (slip) plane.
    Returns (mesh_obj, model, transverse); transverse is the
    TransverseFarfield for "riemann2" (attach it to the time integrator),
    else None.
    """
    if farfield not in ("split", "riemann", "riemann2"):
        raise ValueError("farfield must be split, riemann or riemann2")
    inner = inner_bdry_function or GEOMETRIES[geometry]["inner"]
    mesh_obj = Mesh(dolfinx_mesh, inner)
    if model == "euler":
        flow = CompressibleEulerModel(mesh_obj, poly_order=0, gamma=gamma)
    elif model in ("sa", "laminar"):
        turb = SpalartAllmarasNeg() if model == "sa" else Laminar()
        flow = CompressibleRANSModel(mesh_obj, Re=Re, turbulence_model=turb, gamma=gamma,
                                     T_inf_dim=T_inf, muscl=muscl, limiter=limiter,
                                     limiter_eps=limiter_eps, jacobian_mode=jacobian_mode)
    else:
        raise ValueError("model must be euler, laminar or sa")
    flow.direct_linear_solver = direct
    if asm_overlap is not None:
        flow.asm_overlap = asm_overlap
    if ilu_levels is not None:
        flow.ilu_levels = ilu_levels
    alpha = np.radians(alpha_deg)
    flow.define_inlet_outlet_conditions({'inlet': {'rho': 1.0, 'M': mach, 'p': 1.0, 'alpha': alpha},
                                         'outlet': {'p': 1.0}})

    comm = dolfinx_mesh.comm
    inner_mask = np.asarray(mesh_obj.inner_bdry_facet_mask, dtype=bool)
    sym_tags = []
    outer = ~inner_mask
    if dolfinx_mesh.topology.dim == 3:
        sym = (mesh_obj.bdry_midpoints[:, 1] <= 1e-6) & ~inner_mask
        sym_tags = [(sym, SYMMETRY_TAG)]
        outer = outer & ~sym
    if farfield == "split":
        x = dolfinx_mesh.geometry.x
        x_mid = 0.5 * (comm.allreduce(x[:, 0].max(), MPI.MAX) + comm.allreduce(x[:, 0].min(), MPI.MIN))
        split = -np.tan(alpha) * mesh_obj.bdry_midpoints[:, 2] + x_mid
        mesh_obj.tag_boundary_facets([((mesh_obj.bdry_midpoints[:, 0] <= split) & outer, INLET_TAG),
                                      ((mesh_obj.bdry_midpoints[:, 0] > split) & outer, OUTLET_TAG),
                                      (inner_mask, WALL_TAG)] + sym_tags)
        flow.alpha_tagging = alpha
    else:
        mesh_obj.tag_boundary_facets([(outer, FARFIELD_TAG), (inner_mask, WALL_TAG)] + sym_tags)
    qd = {'quadrature_degree': quadrature_degree}
    mesh_obj.form_manager.build_boundary_measure(measure_metadata=qd)
    mesh_obj.form_manager.build_cell_and_interior_facet_measures(measure_metadata=qd)

    flow.define_elements_functionspaces()
    flow.define_trial_testfunctions()
    flow.compute_initial_conditions_from_inlet()
    flow.interpolate_solution_vector()
    transverse = None
    if farfield == "split":
        flow.define_subsonic_inflow_bc(INLET_TAG)
        flow.define_subsonic_outflow_bc(OUTLET_TAG)
    else:
        if farfield == "riemann2":
            transverse = TransverseFarfield(flow, FARFIELD_TAG, beta=ff_beta,
                                            relax_length=ff_relax_length)
        flow.define_farfield_bc(FARFIELD_TAG, transverse=transverse)
    flow.define_wall_bc(WALL_TAG)
    if sym_tags:
        flow.define_slipwall_bc(SYMMETRY_TAG)
    flow.compute_weakform()
    return mesh_obj, flow, transverse


class ForceCoefficients:
    """
    c_l, c_d, the friction part of c_d, and c_m from the model's wall force
    density (pressure, plus viscous traction for a viscous model). 2D:
    S_ref = c_ref = 1. 3D (half wing, y spanwise): as DG_postprocessor, S_ref
    is half the wall's z-projected area and c_ref = S_ref / span.
    """

    def __init__(self, model, mesh_obj, alpha_deg, wall_tag=WALL_TAG, aero_centre=None):
        msh = mesh_obj.mesh
        comm = msh.comm
        self.model, self.comm = model, comm
        dim = msh.topology.dim
        gamma = model.gamma
        inlet = model.boundary_conditions['inlet']
        q_inf = 0.5 * inlet['rho'] * (inlet['M'] * inlet['c']) ** 2
        n = mesh_obj.n
        ds = mesh_obj.form_manager.ds(wall_tag)
        drag_vec = model._inflow_direction_numpy(np.radians(alpha_deg))
        if dim == 2:
            lift_vec = np.array([-drag_vec[1], drag_vec[0]])
            S_ref, c_ref = 1.0, 1.0
            centre = aero_centre or (0.25, 0.0)
        else:
            lift_vec = np.cross(drag_vec, [0.0, 1.0, 0.0])
            S_ref = 0.5 * comm.allreduce(dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(abs(n[2]) * ds)), MPI.SUM)
            bbox = mesh_obj.compute_wall_bounding_box(wall_tag)
            c_ref = S_ref / (bbox[1][1] - bbox[1][0])
            centre = aero_centre or (0.25, 0.0, 0.0)
        u = model.u_vec
        force = model.wall_force_density(u, n)
        friction = force - pressure(model.mean_flow(u), gamma) * n
        lever = ufl.SpatialCoordinate(msh) - ufl.as_vector(list(centre))
        qS = q_inf * S_ref
        self.S_ref, self.c_ref = S_ref, c_ref
        self._forms = [dolfinx.fem.form(ufl.dot(force, ufl.as_vector(lift_vec)) / qS * ds),
                       dolfinx.fem.form(ufl.dot(force, ufl.as_vector(drag_vec)) / qS * ds),
                       dolfinx.fem.form(ufl.dot(friction, ufl.as_vector(drag_vec)) / qS * ds),
                       dolfinx.fem.form(moment_integrand(force, lever, dim) / (qS * c_ref) * ds)]

    def __call__(self):
        """(c_l, c_d, c_d_friction, c_m) of the current state. Collective."""
        self.model.update_derived_fields()
        return tuple(self.comm.allreduce(dolfinx.fem.assemble_scalar(f), MPI.SUM) for f in self._forms)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", required=True,
                   help="XDMF mesh, or a structured CGNS grid one cell thick (2D airfoil)")
    p.add_argument("--mesh-name", default=None, help="XDMF grid name (default per geometry)")
    p.add_argument("--geometry", choices=sorted(GEOMETRIES), default="airfoil")
    p.add_argument("--model", choices=("euler", "laminar", "sa"), default="sa")
    p.add_argument("--mach", type=float, default=0.7)
    p.add_argument("--alpha", type=float, default=2.0, help="angle of attack [deg]")
    p.add_argument("--Re", type=float, default=6e6)
    p.add_argument("--T-inf", type=float, default=300.0, help="free-stream temperature [K] (Sutherland)")
    p.add_argument("--farfield", choices=("split", "riemann", "riemann2"), default="split")
    p.add_argument("--ff-beta", type=float, default=0.5, help="riemann2 transverse weight")
    p.add_argument("--ff-relax-length", type=float, default=50.0, help="riemann2 relaxation length")
    # numerics
    p.add_argument("--no-muscl", action="store_true", help="first-order RANS inviscid flux")
    p.add_argument("--limiter", choices=("none", "vanalbada"), default="none")
    p.add_argument("--limiter-eps", type=float, default=1e-2)
    p.add_argument("--jacobian", choices=("compact_pc", "exact", "compact"), default="compact_pc")
    p.add_argument("--direct", action="store_true", help="MUMPS for every linear solve")
    p.add_argument("--asm-overlap", type=int, default=None)
    p.add_argument("--ilu-levels", type=int, default=None)
    # steady
    p.add_argument("--cfl0", type=float, default=5.0)
    p.add_argument("--steady-tol", type=float, default=1e-9)
    p.add_argument("--max-steps", type=int, default=1000, help="steady: PTC step budget")
    # unsteady
    p.add_argument("--unsteady", action="store_true")
    p.add_argument("--dual-time", action="store_true", help="PTC inside every physical step")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--dt-start", type=float, default=1e-3)
    p.add_argument("--dt-growth", type=float, default=1.5)
    p.add_argument("--max-retries", type=int, default=8)
    p.add_argument("--order", type=int, choices=(1, 2), default=2)
    p.add_argument("--newton-rtol", type=float, default=1e-8)
    p.add_argument("--newton-atol", type=float, default=1e-10)
    p.add_argument("--newton-max-it", type=int, default=20)
    p.add_argument("--restart", default=None, help="checkpoint .npz to continue from")
    # output
    p.add_argument("--out-dir", default=".")
    p.add_argument("--label", default="")
    p.add_argument("--write-every", type=int, default=50, help="unsteady: wall CSV/VTX every N steps")
    p.add_argument("--write-fields", action="store_true", help="VTX solution, pressure, Mach")
    p.add_argument("--checkpoint-every", type=int, default=500)
    return p.parse_args()


def wall_writer(model, mesh_obj):
    """write(path): wall C_p and c_f at both ends and the middle of every 2D
    wall facet, evaluated from the facet's own cell (the two values a wall node
    gets from its two facets differ by the interelement jump)."""
    msh = mesh_obj.mesh
    comm = msh.comm
    if msh.topology.dim != 2:
        return lambda path: None
    fdim = 1
    inlet = model.boundary_conditions['inlet']
    q_inf = 0.5 * inlet['rho'] * (inlet['M'] * inlet['c']) ** 2
    u, n = model.u_vec, mesh_obj.n
    cp = (pressure(model.mean_flow(u), model.gamma) - inlet['p']) / q_inf
    fields = [ufl.SpatialCoordinate(msh)[0], ufl.SpatialCoordinate(msh)[1], cp]
    if hasattr(model, "skin_friction_vector"):
        drag_dir = ufl.as_vector(list(model._inflow_direction_numpy(inlet['alpha'])))
        fields.append(ufl.dot(model.skin_friction_vector(u, n, q_inf), drag_dir))
    facets = mesh_obj.meshtags.find(WALL_TAG)
    facets = facets[facets < msh.topology.index_map(fdim).size_local]
    f2c = msh.topology.connectivity(fdim, 2)
    c2f = msh.topology.connectivity(2, fdim)
    ent = np.array([(f2c.links(f)[0], int(np.where(c2f.links(f2c.links(f)[0]) == f)[0][0]))
                    for f in facets], dtype=np.int32).reshape(-1, 2)
    s = np.array([[0.0], [0.5], [1.0]])
    exprs = [dolfinx.fem.Expression(e, s, comm=comm) for e in fields]

    def write(path):
        model.update_derived_fields()
        cols = [e.eval(msh, ent).reshape(-1, 3) if len(ent) else np.zeros((0, 3)) for e in exprs]
        if len(cols) == 3:   # inviscid: no skin friction
            cols.append(np.zeros_like(cols[0]))
        rows = [np.column_stack([np.full(3, f), s[:, 0], *[c[i] for c in cols]])
                for i, f in enumerate(facets)]
        data = comm.gather(np.vstack(rows) if rows else np.zeros((0, 6)), root=0)
        if comm.rank == 0:
            np.savetxt(path, np.vstack(data), delimiter=",", comments="", header="facet,s,x,y,cp,cf")
    return write


def field_writer(model, directory, prefix=""):
    """write(t=None): VTX files (<prefix>solution, pressure, mach .bp) of the
    current state in `directory`, one time point per call. Collective."""
    comm = model.u_vec.function_space.mesh.comm
    Vs = model.functionspaces["V_scalar"]
    p_f, M_f = dolfinx.fem.Function(Vs, name="pressure"), dolfinx.fem.Function(Vs, name="mach")
    U = model.mean_flow(model.u_vec)
    p_e = pressure(U, model.gamma)
    vel = ufl.as_vector([U[1 + i] / U[0] for i in range(model.dimensions)])
    M_e = ufl.sqrt(ufl.dot(vel, vel)) / ufl.sqrt(model.gamma * p_e / U[0])
    pts = Vs.element.interpolation_points
    exprs = [(p_f, dolfinx.fem.Expression(p_e, pts)), (M_f, dolfinx.fem.Expression(M_e, pts))]
    writers = [(FileWriter(os.path.join(directory, prefix + name), comm, f.function_space), f)
               for name, f in (("solution", model.u_vec), ("pressure", p_f), ("mach", M_f))]

    def write(t=None):
        for f, e in exprs:
            f.interpolate(e)
        for w, f in writers:
            w.interpolate_and_write(f, write_counter=t)
    return write


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    geo = GEOMETRIES[args.geometry]
    dolfinx_mesh, _ = load_dolfinx_mesh(args.mesh, comm, mesh_name=args.mesh_name or geo["mesh_name"])
    if args.farfield == "riemann2" and not args.unsteady:
        raise SystemExit("--farfield riemann2 is for unsteady runs only")

    mesh_obj, model, transverse = build_model(
        dolfinx_mesh, model=args.model, mach=args.mach, alpha_deg=args.alpha, Re=args.Re,
        geometry=args.geometry, farfield=args.farfield, T_inf=args.T_inf,
        asm_overlap=args.asm_overlap, ilu_levels=args.ilu_levels, direct=args.direct,
        muscl=not args.no_muscl, limiter=None if args.limiter == "none" else args.limiter,
        limiter_eps=args.limiter_eps, jacobian_mode=args.jacobian, ff_beta=args.ff_beta,
        ff_relax_length=args.ff_relax_length)
    model.report_positivity_limiter = False

    # run folder: unique, created by rank 0
    run_dir = None
    if comm.rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        stem = "{}_{}_{}_M{}_a{}{}_".format("unsteady" if args.unsteady else "steady", args.model,
                                           args.farfield, args.mach, args.alpha,
                                           "_" + args.label if args.label else "")
        run_dir = tempfile.mkdtemp(prefix=stem + time.strftime("%Y%m%d-%H%M%S_"), dir=args.out_dir)
    run_dir = comm.bcast(run_dir, root=0)

    config = {"mesh": os.path.abspath(args.mesh), "mach": args.mach, "alpha": args.alpha,
              "Re": args.Re, "model": args.model, "farfield": args.farfield}
    coefficients = ForceCoefficients(model, mesh_obj, args.alpha, aero_centre=geo["aero_centre"])
    write_wall = wall_writer(model, mesh_obj)
    write_fields = field_writer(model, run_dir) if args.write_fields else None

    def write(tag, t=None):
        write_wall(os.path.join(run_dir, "wall_{}.csv".format(tag)))
        if write_fields:
            write_fields(t)

    PETSc.Sys.Print("=" * 72)
    PETSc.Sys.Print("{} {} p=0: {} cells, {} dofs, {} ranks, M={}, alpha={} deg{}, far field {}".format(
        "Unsteady" if args.unsteady else "Steady", args.model,
        dolfinx_mesh.topology.index_map(dolfinx_mesh.topology.dim).size_global,
        model.u_vec.x.petsc_vec.getSize(), comm.size, args.mach, args.alpha,
        "" if args.model == "euler" else ", Re={:g}".format(args.Re), args.farfield))
    PETSc.Sys.Print("Output: {}".format(run_dir))
    PETSc.Sys.Print("=" * 72)

    if not args.unsteady:
        model.use_ptc = True
        model.ptc_settings = {**model.ptc_settings, "cfl0": args.cfl0}
        model.max_newton_iterations = args.max_steps
        model.residual_conv_limit = args.steady_tol
        model.set_up_solver()
        tic = time.perf_counter()
        r = model.solve_system().norm()
        wall = time.perf_counter() - tic
        steps = model.time_integrator.last_steady_steps
        cl, cd, cdf, cm = coefficients()
        write("final")
        if comm.rank == 0:
            np.savetxt(os.path.join(run_dir, "result.csv"),
                       np.array([[float(model.solver_converged), steps, r, wall, cl, cd, cdf, cm]]),
                       delimiter=",", comments="", header="converged,steps,res,wall_s,c_l,c_d,c_d_friction,c_m")
        save_checkpoint(os.path.join(run_dir, "checkpoint_final.npz"), [model.u_vec],
                        {**config, "steady": True})
        PETSc.Sys.Print("RESULT steady converged={} steps={} |R|={:.3e} c_l={:.6f} c_d={:.6f} "
                        "(friction {:.6f}) c_m={:.6f}  [{:.1f} s]".format(
                            model.solver_converged, steps, r, cl, cd, cdf, cm, wall))
        PETSc.Sys.Print("Done. Results in {}".format(run_dir))
        return

    # ------------------------------------------------------------------
    # unsteady
    # ------------------------------------------------------------------
    integrator = model.enable_time_stepping(dual_time=args.dual_time)
    model.set_up_solver()
    if transverse is not None:
        transverse.attach(integrator)
    term = integrator.term
    ckpt_fields = [term.U_n, term.U_nm1] + ([] if transverse is None else transverse.functions)
    step0, t0, dt_prev = 0, 0.0, None
    if args.restart:
        meta = load_checkpoint(args.restart, ckpt_fields, optional=True)
        bad = {k: (meta.get(k), config[k]) for k in RESTART_KEYS if meta.get(k) != config[k]}
        if bad:
            raise ValueError("restart settings differ from the checkpoint: {}".format(bad))
        step0, t0 = meta["step"], meta["t"]
        dt_prev = meta.get("dt_last") if step0 >= 1 else None
        model.u_vec.x.array[:] = term.U_n.x.array
        model.u_vec.x.scatter_forward()
        PETSc.Sys.Print("Restarted from {} (step {}, t={})".format(args.restart, step0, t0))
        U_nm1 = term.U_nm1.x.array.copy()
        integrator.start(t0, step0, dt_prev)
        term.U_nm1.x.array[:] = U_nm1
    else:
        integrator.start(0.0, 0, None)
    control = StepSizeController(dt=args.dt, dt_start=args.dt_start, growth=args.dt_growth,
                                 max_retries=args.max_retries)
    dual = CFLController(**model.ptc_settings) if args.dual_time else None

    header = "step,t,dt,newton_its,ksp_its,res_initial,res_final,wall_s,c_l,c_d,c_d_friction,c_m"
    history = [(step0, t0, np.nan, 0, 0, np.nan, np.nan, 0.0, *coefficients())]
    write("{:06d}".format(step0), t0)

    def save_history():
        if comm.rank == 0:
            np.savetxt(os.path.join(run_dir, "history.csv"), np.array(history), delimiter=",",
                       comments="", header=header)

    def checkpoint(name):
        save_checkpoint(os.path.join(run_dir, name), ckpt_fields,
                        {**config, "step": integrator.step_index, "t": integrator.time,
                         "dt_last": integrator.dt_prev})

    failed = False
    for k in range(1, args.steps + 1):
        tic = time.perf_counter()
        rec = integrator.advance(control, order=args.order, newton_rtol=args.newton_rtol,
                                 newton_atol=args.newton_atol, newton_max_it=args.newton_max_it,
                                 dual_time=dual)
        wall = time.perf_counter() - tic
        if not rec.accepted:
            PETSc.Sys.Print("STOP: step {} failed after {} halvings".format(
                integrator.step_index + 1, args.max_retries))
            failed = True
            break
        cl, cd, cdf, cm = coefficients()
        step, t = integrator.step_index, integrator.time
        history.append((step, t, rec.dt, rec.newton_its, rec.ksp_its, rec.res_initial,
                        rec.res_final, wall, cl, cd, cdf, cm))
        PETSc.Sys.Print(
            "step {:4d}  t={:8.4f}  dt={:.3e}  BDF{}  Newton {:2d}  KSP {:4d}  |R| {:.3e} -> {:.3e}  "
            "c_l {:+.6f}  c_d {:+.6f} (fric {:+.6f})  c_m {:+.6f}  [{:.1f} s]".format(
                step, t, rec.dt, rec.order, rec.newton_its, rec.ksp_its, rec.res_initial,
                rec.res_final, cl, cd, cdf, cm, wall))
        if k % args.write_every == 0 or k == args.steps:
            write("{:06d}".format(step), t)
            save_history()
        if args.checkpoint_every and k % args.checkpoint_every == 0 and k < args.steps:
            checkpoint("checkpoint_latest.npz")
    save_history()
    checkpoint("checkpoint_final.npz")
    PETSc.Sys.Print("{} Results in {}".format("FAILED." if failed else "Done.", run_dir))


if __name__ == "__main__":
    main()
