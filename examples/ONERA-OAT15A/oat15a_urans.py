"""
Forward unsteady RANS (SA-neg, p = 0) of ONERA's OAT15A airfoil on one Rizzi
grid: M 0.73, Re 3e6 (chord), static temperature 271 K, by default Rizzi grid 8
(667,648 cells) at alpha = 10 deg. The flow starts impulsively from the free
stream. A pseudo-transient continuation (--init-steps) first brings it to a
(partly converged) steady state; at this angle of attack, with the flow
separated, it need not converge, and the time-accurate run goes on from
wherever it stops. An impulsive start (--init-steps 0) is stiff: on grid 3 its
first step needed dt* = 1.5e-4. The flow is then advanced with BDF2 (BDF1 for
the first step), a Newton solve per step, a time-step ramp, and step halving on
a failed step; optionally with dual time stepping. The solver is that of
../flow_analysis.py --unsteady.

Times are in convective units, t* = t U_inf / c (c = 1 in the mesh): the
buffet of this airfoil has a period of about 15 t*. --dt and --t-end are in t*.

    OMP_NUM_THREADS=1 mpirun -n 8 python oat15a_urans.py --out-dir <output folder>
    # the same under the external memory guard (recommended for grid 8)
    python ../memory_guard.py --max-gib 10 -- \\
        mpirun -n 8 python oat15a_urans.py --out-dir <output folder>
    # a short smoke test on the 63,488-cell grid 3
    OMP_NUM_THREADS=1 mpirun -n 4 python oat15a_urans.py --grid 3 --t-end 0.5 --dt 0.05

Memory: the run checks its own memory after every time step (Pss summed over
the ranks, see ../memory_guard.py) and the machine's available memory. If the
ranks together use more than --max-mem-gib (default: 60 % of the machine's
memory) or less than --min-available-gib is left, it writes a checkpoint and
exits with status 3; --restart continues from it. A spike inside one time step
is outside its reach: wrap the run in ../memory_guard.py for that.

Every run writes into its own new folder under --out-dir:

  config.json            the settings
  history.csv            one row per accepted time step: step, t, t*, dt, Newton
                         and Krylov iterations, residual, wall time, c_l, c_d,
                         the friction part of c_d, c_m, memory of the ranks [GiB]
  wall_<step>.csv        C_p and c_f at the ends and middle of every wall facet,
                         every --write-every steps
  checkpoint_init.npz    the state after the initialisation (time 0)
  checkpoint_latest.npz  every --checkpoint-every steps; checkpoint_final.npz at
                         the end (partition independent; restart with --restart)
  solution/pressure/mach .bp   VTX fields every --write-every steps, with
                         --write-fields

The model set-up, force coefficients and wall output are those of
../flow_analysis.py.
"""
import argparse
import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dragonfly_sim.utils.mesh_io_utils import load_dolfinx_mesh
from dragonfly_sim.utils.checkpoint import save_checkpoint, load_checkpoint
from dragonfly_sim.core.time_integration import StepSizeController, CFLController

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
from flow_analysis import build_model, ForceCoefficients, wall_writer, field_writer  # noqa: E402
from oat15a_analysis import MESH_DIR, count_wall_facets  # noqa: E402
from memory_guard import process_gib, available_gib, total_gib, KILLED_STATUS  # noqa: E402

INIT_CHUNK = 10     # pseudo-transient steps between memory checks
RESTART_KEYS =("grid", "mach", "alpha", "Re", "T_inf", "farfield")
RIZZI_DIR = "ONERA-ONERA-OAT15A-Rizzi"
HISTORY_HEADER = "step,t,t_conv,dt,newton_its,ksp_its,res_initial,res_final,wall_s,c_l,c_d,c_d_friction,c_m,mem_gib"


def rizzi_grid_path(mesh_dir, level):
    """The Rizzi grid file of `level`: where meshes/README.md says it is, or one
    folder deeper, where unzipping the download puts it."""
    name = os.path.join("grid{0}", "OAT15A_Rizzi_{0}.cgns").format(level)
    candidates = [os.path.join(mesh_dir, RIZZI_DIR, name), os.path.join(mesh_dir, RIZZI_DIR, RIZZI_DIR, name)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("Rizzi grid {} not found; looked for:\n  {}".format(level, "\n  ".join(candidates)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, default=8, help="Rizzi grid level (1-9)")
    p.add_argument("--mesh", default=None, help="CGNS grid file, instead of --grid")
    p.add_argument("--mesh-dir", default=MESH_DIR)
    p.add_argument("--alpha", type=float, default=10.0, help="angle of attack [deg]")
    p.add_argument("--mach", type=float, default=0.73)
    p.add_argument("--Re", type=float, default=3e6, help="Reynolds number based on the chord")
    p.add_argument("--T-inf", type=float, default=271.0,
                   help="static free-stream temperature [K] (Sutherland); DPW8 test case 1a: 271 K")
    p.add_argument("--farfield", choices=("split", "riemann", "riemann2"), default="riemann",
                   help="riemann: characteristic far field on the whole outer boundary; split: steady-style "
                        "inflow/outflow; riemann2: riemann with the transverse non-reflecting correction "
                        "(core/farfield.py), which at present fails in its set-up on the OAT15A CGNS grids "
                        "(\"cannot match facet quadrature points\"), serial or parallel")
    p.add_argument("--ff-beta", type=float, default=0.5, help="riemann2 transverse weight")
    p.add_argument("--ff-relax-length", type=float, default=50.0, help="riemann2 relaxation length [chords]")
    p.add_argument("--limiter", choices=("none", "vanalbada"), default="none",
                   help="slope limiter of the MUSCL reconstruction")
    p.add_argument("--limiter-eps", type=float, default=1e-2, help="Van Albada smoothness parameter")
    # initial state
    p.add_argument("--init-steps", type=int, default=300,
                   help="pseudo-transient continuation steps from the free stream before the time-accurate run "
                        "(0: impulsive start, which at this angle of attack needs time steps of 1e-4 t* at first). "
                        "The run goes on from the state reached, converged or not. Ignored with --restart")
    p.add_argument("--init-tol", type=float, default=1e-8, help="residual at which the initialisation stops")
    p.add_argument("--cfl0", type=float, default=5.0, help="initial CFL of the initialisation (and of dual time)")
    # time stepping, in convective units t* = t U_inf / c
    p.add_argument("--t-end", type=float, default=100.0, help="end time [t*]; ~6 buffet periods")
    p.add_argument("--dt", type=float, default=0.02, help="time step [t*]")
    p.add_argument("--dt-start", type=float, default=None, help="first time step [t*] (default dt/20)")
    p.add_argument("--dt-growth", type=float, default=1.5, help="time-step growth per step up to dt")
    p.add_argument("--max-retries", type=int, default=8, help="step halvings before giving up")
    p.add_argument("--max-steps", type=int, default=100000, help="time-step budget of this run")
    p.add_argument("--order", type=int, choices=(1, 2), default=2, help="BDF order")
    p.add_argument("--dual-time", action="store_true", help="pseudo-time continuation inside every step")
    p.add_argument("--newton-criterion", choices=("incremental", "residual"), default="incremental",
                   help="Newton convergence test: the size of the update dx (default), or the residual |R*|. The "
                        "residual is volume weighted and carries rounding error of ~1e-16 |M U| / dt, with |M U| "
                        "~ 1e9 on these grids (huge far-field cells): above any useful absolute tolerance, and "
                        "larger for the smaller steps a failed step retries with. Ignored with --dual-time")
    p.add_argument("--newton-rtol", type=float, default=1e-6,
                   help="relative tolerance (to the first update, or to |R*_0| with --newton-criterion residual)")
    p.add_argument("--newton-atol", type=float, default=1e-9,
                   help="absolute tolerance, in state units for the update (use ~1e-4 with the residual criterion)")
    p.add_argument("--newton-max-it", type=int, default=20)
    p.add_argument("--restart", default=None, help="checkpoint .npz to continue from")
    # memory
    p.add_argument("--max-mem-gib", type=float, default=None,
                   help="stop cleanly when the ranks together use more than this [GiB] "
                        "(default: 60%% of the machine's memory)")
    p.add_argument("--min-available-gib", type=float, default=1.5,
                   help="stop cleanly when the machine's available memory falls below this [GiB]")
    # output
    p.add_argument("--out-dir", default=".")
    p.add_argument("--label", default="")
    p.add_argument("--write-every", type=int, default=100, help="wall CSV (and VTX fields) every N steps")
    p.add_argument("--write-fields", action="store_true", help="VTX solution, pressure and Mach fields")
    p.add_argument("--checkpoint-every", type=int, default=250)
    return p.parse_args()


def memory_state(comm):
    """(Pss of all ranks in GiB, smallest MemAvailable of the ranks' nodes in GiB). Collective."""
    return comm.allreduce(process_gib(), MPI.SUM), comm.allreduce(available_gib(), MPI.MIN)


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    if args.max_mem_gib is None:
        args.max_mem_gib = 0.6 * total_gib()
    mesh_path = args.mesh or rizzi_grid_path(args.mesh_dir, args.grid)
    dt_start = args.dt_start if args.dt_start is not None else args.dt / 20

    run_dir = None
    if comm.rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        stem = "oat15a_urans_grid{}_sa_M{}_a{}{}_".format(args.grid, args.mach, args.alpha,
                                                         "_" + args.label if args.label else "")
        run_dir = tempfile.mkdtemp(prefix=stem + time.strftime("%Y%m%d-%H%M%S_"), dir=args.out_dir)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({**vars(args), "mesh_path": os.path.abspath(mesh_path), "ranks": comm.size}, f, indent=2)
    run_dir = comm.bcast(run_dir, root=0)

    msh, info = load_dolfinx_mesh(mesh_path, comm)
    PETSc.Sys.Print("grid {}: {} cells, {} wall edges, chord {} (file units)".format(
        args.grid, info["n_cells"], info["n_wall_edges"], info["chord"]))
    mesh_obj, model, transverse = build_model(
        msh, model="sa", mach=args.mach, alpha_deg=args.alpha, Re=args.Re, T_inf=args.T_inf,
        farfield=args.farfield, limiter=None if args.limiter == "none" else args.limiter,
        limiter_eps=args.limiter_eps, ff_beta=args.ff_beta, ff_relax_length=args.ff_relax_length)
    n_wall = count_wall_facets(mesh_obj)
    if n_wall != info["n_wall_edges"]:
        raise RuntimeError("the solver's wall has {} facets, the CGNS Wall family {} edges"
                           .format(n_wall, info["n_wall_edges"]))
    model.report_positivity_limiter = False
    model.ptc_settings = {**model.ptc_settings, "cfl0": args.cfl0}

    # model time units: rho = p = 1, so the free-stream speed is M c_inf and the chord is 1
    inlet = model.boundary_conditions['inlet']
    U_inf = inlet['M'] * inlet['c']
    t_end, dt, dt_start = args.t_end / U_inf, args.dt / U_inf, dt_start / U_inf

    coefficients = ForceCoefficients(model, mesh_obj, args.alpha, aero_centre=(0.25, 0.0))
    write_wall = wall_writer(model, mesh_obj)
    write_fields = field_writer(model, run_dir) if args.write_fields else None

    def write(tag, t=None):
        write_wall(os.path.join(run_dir, "wall_{}.csv".format(tag)))
        if write_fields:
            write_fields(t)

    mem0, avail0 = memory_state(comm)
    PETSc.Sys.Print("=" * 72)
    PETSc.Sys.Print("Unsteady SA RANS p=0: {} cells, {} dofs, {} ranks, M={}, Re={:g}, alpha={} deg, far field {}"
                    .format(info["n_cells"], model.u_vec.x.petsc_vec.getSize(), comm.size, args.mach, args.Re,
                            args.alpha, args.farfield))
    PETSc.Sys.Print("t* = t U_inf = {:.4f} t; dt = {} t*, t_end = {} t*, BDF{}{}".format(
        1.0 / U_inf, args.dt, args.t_end, args.order, ", dual time" if args.dual_time else ""))
    PETSc.Sys.Print("Memory: {:.2f} GiB in use by the ranks, {:.2f} GiB available; stop at > {:.2f} GiB or "
                    "< {:.2f} GiB available".format(mem0, avail0, args.max_mem_gib, args.min_available_gib))
    PETSc.Sys.Print("Output: {}".format(run_dir))
    PETSc.Sys.Print("=" * 72)

    # use_ptc and the physical time term together give one integrator that does both the initial
    # pseudo-transient continuation (physical term off) and the time steps (pseudo term off)
    initialise = args.init_steps > 0 and not args.restart
    model.use_ptc = initialise or args.dual_time
    integrator = model.enable_time_stepping(dual_time=args.dual_time)
    model.set_up_solver()
    if initialise:
        model.residual_conv_limit = args.init_tol
        # in chunks, with one CFL controller throughout, to check the memory between them
        ctrl, steps, tic = CFLController(**model.ptc_settings), 0, time.perf_counter()
        while steps < args.init_steps:
            integrator.solve_steady(controller=ctrl, max_steps=min(INIT_CHUNK, args.init_steps - steps))
            steps += integrator.last_steady_steps
            mem, avail = memory_state(comm)
            if model.solver_converged or ctrl.exhausted:
                break
            if mem > args.max_mem_gib or avail < args.min_available_gib:
                PETSc.Sys.Print("STOP: memory ({:.2f} GiB in use by the ranks, limit {:.2f}; {:.2f} GiB available, "
                                "minimum {:.2f}) during the initialisation, after {} steps".format(
                                    mem, args.max_mem_gib, avail, args.min_available_gib, steps))
                return KILLED_STATUS
        PETSc.Sys.Print("INIT pseudo-transient continuation: {} steps, converged={}, |R|={:.3e}, c_l {:+.6f}, "
                        "c_d {:+.6f}  [{:.1f} s, {:.2f} GiB, {:.2f} GiB avail]".format(
                            steps, bool(model.solver_converged), model.residual_vec_physical.norm(),
                            *coefficients()[:2], time.perf_counter() - tic, mem, avail))
    if transverse is not None:      # after the steady solve, which has no time-dependent boundaries
        transverse.attach(integrator)
    term = integrator.term
    ckpt_fields = [term.U_n, term.U_nm1] + ([] if transverse is None else transverse.functions)
    config = {"grid": args.grid if args.mesh is None else args.mesh, "mach": args.mach, "alpha": args.alpha,
              "Re": args.Re, "T_inf": args.T_inf, "farfield": args.farfield}
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
        PETSc.Sys.Print("Restarted from {} (step {}, t*={:.4f})".format(args.restart, step0, t0 * U_inf))
        U_nm1 = term.U_nm1.x.array.copy()
        integrator.start(t0, step0, dt_prev)
        term.U_nm1.x.array[:] = U_nm1
    else:
        integrator.start(0.0, 0, None)
    control = StepSizeController(dt=dt, dt_start=dt_start, growth=args.dt_growth, max_retries=args.max_retries)
    dual = CFLController(**model.ptc_settings) if args.dual_time else None

    history = [(step0, t0, t0 * U_inf, np.nan, 0, 0, np.nan, np.nan, 0.0, *coefficients(), mem0)]
    write("{:06d}".format(step0), t0)

    def save_history():
        if comm.rank == 0:
            np.savetxt(os.path.join(run_dir, "history.csv"), np.array(history), delimiter=",",
                       comments="", header=HISTORY_HEADER)

    def checkpoint(name):
        save_checkpoint(os.path.join(run_dir, name), ckpt_fields,
                        {**config, "step": integrator.step_index, "t": integrator.time,
                         "dt_last": integrator.dt_prev})

    if initialise:      # the start of the time-accurate run, to restart from without repeating the initialisation
        checkpoint("checkpoint_init.npz")

    status, k = 0, 0
    while integrator.time < t_end * (1 - 1e-12) and k < args.max_steps:
        k += 1
        tic = time.perf_counter()
        rec = integrator.advance(control, order=args.order, newton_rtol=args.newton_rtol,
                                 newton_atol=args.newton_atol, newton_max_it=args.newton_max_it,
                                 dual_time=dual, newton_criterion=args.newton_criterion)
        wall = time.perf_counter() - tic
        if not rec.accepted:
            PETSc.Sys.Print("STOP: step {} failed after {} halvings".format(
                integrator.step_index + 1, args.max_retries))
            status = 1
            break
        cl, cd, cdf, cm = coefficients()
        mem, avail = memory_state(comm)
        step, t = integrator.step_index, integrator.time
        history.append((step, t, t * U_inf, rec.dt, rec.newton_its, rec.ksp_its, rec.res_initial,
                        rec.res_final, wall, cl, cd, cdf, cm, mem))
        PETSc.Sys.Print(
            "step {:5d}  t*={:8.4f}  dt*={:.3e}  BDF{}  Newton {:2d}  KSP {:4d}  |R| {:.3e} -> {:.3e}  "
            "c_l {:+.6f}  c_d {:+.6f} (fric {:+.6f})  c_m {:+.6f}  [{:.1f} s, {:.2f} GiB, {:.2f} GiB avail]".format(
                step, t * U_inf, rec.dt * U_inf, rec.order, rec.newton_its, rec.ksp_its, rec.res_initial,
                rec.res_final, cl, cd, cdf, cm, wall, mem, avail))
        if not np.all(np.isfinite([cl, cd, cm])):
            PETSc.Sys.Print("STOP: non-finite force coefficients")
            status = 1
            break
        if mem > args.max_mem_gib or avail < args.min_available_gib:
            PETSc.Sys.Print("STOP: memory ({:.2f} GiB in use by the ranks, limit {:.2f}; {:.2f} GiB available, "
                            "minimum {:.2f}). Checkpointing; continue with --restart.".format(
                                mem, args.max_mem_gib, avail, args.min_available_gib))
            status = KILLED_STATUS
            break
        if k % args.write_every == 0:
            write("{:06d}".format(step), t)
            save_history()
        if args.checkpoint_every and k % args.checkpoint_every == 0:
            checkpoint("checkpoint_latest.npz")
    save_history()
    if integrator.step_index > step0:
        write("{:06d}".format(integrator.step_index), integrator.time)
        checkpoint("checkpoint_final.npz")
    PETSc.Sys.Print("{} Results in {}".format(
        "Done." if status == 0 else "STOPPED (status {}).".format(status), run_dir))
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        # an exception on one rank would leave the others waiting for it in a collective call, forever
        traceback.print_exc()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
