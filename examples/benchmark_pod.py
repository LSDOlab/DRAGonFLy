"""
POD reduced-order model benchmark: full-order (FOM) against reduced-order
(ROM) solutions of the Euler and the RANS model at the same design points.

Each design point is a shape (FFD control-point motions), an angle of attack
and a Mach number. For each flow model in --models the script builds its own
DG_windtunnel_model (with no reduced-order model of its own, so every solve
through it is a FOM solve) and one core.reduced_order_model.ReducedOrderModel
per POD variant in --variants:

  pod_global   the leading rb_size POD modes of all training snapshots
  pod_local    the local_weighted basis, recomputed for each test point
               from the distance-weighted snapshots (Cubic weights; see
               --cubic-cutoff and --n-nonzero-weights)

Workflow (per flow model, on the same samples):
  1. Draw a Latin-Hypercube training set over (shape, alpha, Mach), or read
     a previously written one (--read-training-data).
  2. Solve the FOM at every training point and add the converged solution as
     a snapshot to every variant's POD.
  3. Draw a separate, uniformly random test set. At each test point solve the
     FOM (the reference), then each variant's LSPG ROM on the same deformed
     mesh, and record the state L2 errors (domain and wall), the lift, drag
     and moment errors, the ROM residual and both wall times.
  4. Write one DataStore per (model, variant) to <output-dir>/Result_dicts,
     print a summary and save comparison plots (Euler and RANS side by side).

Every setting below is a command-line option; the defaults are those of the
project-sandbox benchmark this script is ported from (p = 0, POD only: the
DEIM/MDEIM hyper-reduction and p > 0 stabilized variants are not part of this
package). Run from any directory, e.g.

    OMP_NUM_THREADS=1 mpirun -n 4 python benchmark_pod.py --dimension 2D --n-train 10 --n-test 5
    OMP_NUM_THREADS=1 mpirun -n 8 python benchmark_pod.py              # 3D wing, Euler and RANS

The 3D RANS solves are expensive (minutes and ~20 GB each on the L3 wing);
--release-solver-memory (on by default in 3D) frees each model's matrices
after every solve, so the Euler and RANS solvers never hold theirs at once.
"""
import argparse
import json
import os
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx
import csdl_alpha as csdl
from scipy.stats import qmc
import matplotlib
matplotlib.use("Agg")  # headless under mpirun: figures are saved, never shown
from matplotlib import pyplot as plt

from dragonfly_sim.core.windtunnel_model import DG_windtunnel_model
from dragonfly_sim.core.RANS_model import CompressibleRANSModel
from dragonfly_sim.core.reduced_order_model import ReducedOrderModel
from dragonfly_sim.core.shape_design import FFDShapeParameterization, ControlPointMotions
from dragonfly_sim.utils.ffd_dv_utils import spanwise_linear_bounds
from dragonfly_sim.utils.mesh_io_utils import load_dolfinx_mesh
from dragonfly_sim.utils.mesh_manager_utils import airfoil_inner_bdry_function, wing_inner_bdry_function
from dragonfly_sim.utils.data_handler import DataStore
from dragonfly_sim.utils.solver_profiling import PROFILER, default_group

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "meshes")
MODELS = ("euler", "rans")
VARIANTS = {"pod_global": "global", "pod_local": "local_weighted"}


# =============================================================================
# Command line
# =============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("problem")
    g.add_argument("--dimension", choices=("2D", "3D"), default="3D",
                   help="2D: NACA0012 airfoil; 3D: the Simple Transonic Wing (L3 grid)")
    g.add_argument("--models", default="euler,rans", help="comma-separated subset of euler,rans")
    g.add_argument("--euler-mesh", default=None, help="Euler mesh (.xdmf/.cgns); default per dimension")
    g.add_argument("--rans-mesh", default=None, help="RANS mesh (.xdmf/.cgns); default per dimension")
    g.add_argument("--reynolds", type=float, default=None,
                   help="RANS Reynolds number per unit mesh length; default 6e6 (2D, chord 1) or "
                        "5e6 on the wing's mean aerodynamic chord (3D)")
    g.add_argument("--mach-lower", type=float, default=0.7)
    g.add_argument("--mach-upper", type=float, default=0.9)
    g.add_argument("--alpha-lower", type=float, default=1.75, help="degrees")
    g.add_argument("--alpha-upper", type=float, default=2.25,
                   help="degrees; the inlet/outlet split is tagged at 2 deg and allows +/-1 deg")
    g.add_argument("--cp-motion-sample-scale", type=float, default=1.0,
                   help="scale of the sampled shape range relative to the design-variable bounds")

    g = p.add_argument_group("sampling")
    g.add_argument("--seed", type=int, default=0, help="training LHS seed; the test seed is seed + 1")
    g.add_argument("--n-train", type=int, default=50)
    g.add_argument("--n-test", type=int, default=25)

    g = p.add_argument_group("full-order solves")
    g.add_argument("--residual-conv-limit", type=float, default=1e-9)
    g.add_argument("--euler-max-newton", type=int, default=100, help="total Newton steps per Euler solve")
    g.add_argument("--rans-max-newton", type=int, default=400, help="pseudo-time steps per RANS solve")
    g.add_argument("--fom-newton-inner-max-it", type=int, default=1,
                   help="Euler Newton steps per residual check (RANS PTC always takes one)")
    g.add_argument("--warm-start", action="store_true",
                   help="start each FOM solve from the previous one (exclusive with Mach continuation)")
    g.add_argument("--no-mach-continuation", dest="mach_continuation", action="store_false",
                   help="start each FOM solve from the free stream instead of the nearest stored "
                        "solution at or below its Mach number")
    g.add_argument("--no-mach-continuation-sort", dest="mach_continuation_sort", action="store_false",
                   help="keep the sampled order instead of processing the samples by ascending Mach")
    g.add_argument("--mach-distance-weight", type=float, default=5.0)
    g.add_argument("--keep-nonconverged", action="store_true",
                   help="keep FOM samples whose solve did not converge")
    g.add_argument("--release-solver-memory", choices=("auto", "on", "off"), default="auto",
                   help="free the solver matrices after every solve (auto: on in 3D)")

    g = p.add_argument_group("POD / ROM")
    g.add_argument("--variants", default="pod_global,pod_local", help="comma-separated subset")
    g.add_argument("--rb-size", type=int, default=10)
    g.add_argument("--max-rank", type=int, default=None, help="stored snapshot rank; default 10 rb_size")
    g.add_argument("--qr-rank-tol", type=float, default=1e-10)
    g.add_argument("--weightfunction", choices=("Cubic", "Linear", "Quadratic", "InverseDistance"),
                   default="Cubic")
    g.add_argument("--n-nonzero-weights", type=int, default=None,
                   help="Cubic: snapshots with a nonzero weight (default rb_size)")
    g.add_argument("--cubic-cutoff", type=float, default=None,
                   help="Cubic: c in [0, 1) of the support radius c d_min + (1 - c) d_max "
                        "(never below the rb_size nearest snapshots); default: n_nonzero_weights")
    g.add_argument("--parameter-distance", choices=("raw", "normalized"), default="raw",
                   help="normalized: divide each parameter by its sampled range in the distance")
    g.add_argument("--inner-product", choices=("euclidean", "l2"), default="euclidean")
    g.add_argument("--state-scaling", choices=("freestream", "none"), default="freestream")
    g.add_argument("--rom-newton-max-it", type=int, default=15)
    g.add_argument("--rom-newton-rtol", type=float, default=1e-10)
    g.add_argument("--rom-newton-atol", type=float, default=1e-12)
    g.add_argument("--rom-max-outer-iterations", type=int, default=3)
    g.add_argument("--rom-g-conv-limit", type=float, default=1e-12)
    g.add_argument("--no-rom-line-search", dest="rom_line_search", action="store_false")
    g.add_argument("--rom-ic", choices=("freestream", "continuation"), default="freestream",
                   help="ROM initial state (projected onto the basis): the free stream, or the FOM's "
                        "initial condition")

    g = p.add_argument_group("run control and output")
    g.add_argument("--read-training-data", action="store_true",
                   help="read the training snapshots written by an earlier run instead of solving")
    g.add_argument("--training-only", action="store_true",
                   help="stop after the training stage (no test points)")
    g.add_argument("--output-dir", default="benchmark_outputs")
    g.add_argument("--no-profiling", dest="profiling", action="store_false",
                   help="skip the per-phase wall-time breakdown of the test solves")
    g.add_argument("--no-profile-barriers", dest="profile_barriers", action="store_false",
                   help="profile without MPI barriers at the phase edges (end-to-end timing "
                        "undisturbed, phase attribution less exact)")
    args = p.parse_args(argv)

    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for m in args.models:
        if m not in MODELS:
            p.error("unknown model {!r} (choose from {})".format(m, ", ".join(MODELS)))
    for v in args.variants:
        if v not in VARIANTS:
            p.error("unknown variant {!r} (choose from {})".format(v, ", ".join(VARIANTS)))
    if args.max_rank is None:
        args.max_rank = 10 * args.rb_size
    if args.cubic_cutoff is not None and not 0.0 <= args.cubic_cutoff < 1.0:
        p.error("--cubic-cutoff must lie in [0, 1)")
    if args.warm_start and args.mach_continuation:
        p.error("--warm-start and Mach continuation are two initial-condition policies; "
                "pass --no-mach-continuation with --warm-start")
    if args.alpha_lower > args.alpha_upper or max(abs(args.alpha_lower - 2.0),
                                                  abs(args.alpha_upper - 2.0)) > 1.0:
        p.error("alpha must stay within 1 deg of the 2 deg the inlet/outlet split is tagged at")
    args.release = (args.release_solver_memory == "on"
                    or (args.release_solver_memory == "auto" and args.dimension == "3D"))
    args.run_tag = "benchmark_pod_{}_rb{}_{}{}{}{}".format(
        args.dimension, args.rb_size, args.inner_product,
        "" if args.state_scaling == "freestream" else "_unscaled",
        "" if args.cubic_cutoff is None else "_c{:g}".format(args.cubic_cutoff),
        "" if args.parameter_distance == "raw" else "_normdist")
    return args


# =============================================================================
# Problem set-up
# =============================================================================
def problem_definition(dimension):
    """The dimension-dependent part of the set-up (sandbox defaults)."""
    if dimension == "2D":
        return dict(
            meshes={"euler": os.path.join(MESH_DIR, "naca0012_euler_mesh_quad_v2.xdmf"),
                    "rans": os.path.join(MESH_DIR, "naca0012_cgrid_rans_coarse.xdmf")},
            tdim=2, mach_0=0.8, reynolds=6e6,
            ffd_shape=[10, 3], ffd_degree=[2, 2], ffd_block_corners=None,
            cp_dv_spec={1: 0.01},
            inner_bdry_function=airfoil_inner_bdry_function,
            aero_center=np.array([0.25, 0.]))
    c_root, taper = 5.0, 0.3
    mac = 2. / 3. * c_root * (1. + taper + taper ** 2) / (1. + taper)
    return dict(
        meshes={"euler": os.path.join(MESH_DIR, "wing_vol_L3.cgns"),
                "rans": os.path.join(MESH_DIR, "wing_vol_L3.cgns")},
        tdim=3, mach_0=0.85, reynolds=5e6 / mac,
        ffd_shape=[4, 5, 3], ffd_degree=[3, 3, 2],
        ffd_block_corners=[[(-0.1, -1e-8, 0.32), (5.1, -1e-8, -0.3)],
                           [(7.4, 14.1, 0.32), (9.1, 14.1, -0.3)]],
        cp_dv_spec={2: spanwise_linear_bounds(base_bound=0.05, scale_at_root=1.0,
                                              scale_at_ref=0.5, y_ref=14.1)},
        inner_bdry_function=wing_inner_bdry_function,
        aero_center=np.array([0.25, 0., 0.]))


class StateNorms:
    """L2 norms of a state array over the domain and over the wall, on the
    model's current (possibly deformed) mesh. Collective."""

    def __init__(self, model, wall_tag):
        self.model = model
        self.f = dolfinx.fem.Function(model.functionspaces["V"])
        e = ufl.inner(self.f, self.f)
        self.domain_form = dolfinx.fem.form(e * model.forms.dx)
        self.wall_form = dolfinx.fem.form(e * model.forms.ds(wall_tag))

    def _norm(self, local, form):
        self.f.x.petsc_vec.array[:] = local
        self.f.x.scatter_forward()
        comm = self.model.mesh_obj.mesh.comm
        return float(np.sqrt(comm.allreduce(dolfinx.fem.assemble_scalar(form), op=MPI.SUM)))

    def errors(self, reference, approximation):
        """(L2, L2 / ||reference||, wall L2, wall L2 / ||reference||_wall)."""
        diff = reference - approximation
        out = []
        for form in (self.domain_form, self.wall_form):
            err, ref = self._norm(diff, form), self._norm(reference, form)
            out += [err, err / ref if ref > 0 else np.nan]
        return tuple(out)


class Problem:
    """One flow model: its windtunnel, CSDL graph, POD variants and records."""

    def __init__(self, kind, args, definition):
        comm = MPI.COMM_WORLD
        self.kind = kind
        self.mesh_path = getattr(args, "{}_mesh".format(kind)) or definition["meshes"][kind]
        mesh, _ = load_dolfinx_mesh(self.mesh_path, comm, tdim=definition["tdim"])
        PETSc.Sys.Print("[{}] mesh {}".format(kind, self.mesh_path))

        boundary_dict = {'inlet': {'rho': 1.0, 'M': definition["mach_0"], 'p': 1.0,
                                   'alpha': np.radians(2.0)},
                         'outlet': {'p': 1.0}}
        self.shape = FFDShapeParameterization(definition["ffd_shape"], definition["ffd_degree"],
                                              definition["ffd_block_corners"],
                                              layers=[ControlPointMotions(definition["cp_dv_spec"])])
        kwargs = dict(asm_overlap=1, ilu_levels=1)
        if kind == "rans":
            self.reynolds = args.reynolds if args.reynolds is not None else definition["reynolds"]
            kwargs = dict(asm_overlap=1, ilu_levels=2, model_class=CompressibleRANSModel,
                          model_kwargs={'Re': self.reynolds})

        recorder = csdl.Recorder(inline=False)
        recorder.start()
        wt = DG_windtunnel_model(mesh, boundary_dict, self.shape, definition["aero_center"],
                                 mesh_inner_bdry_function=definition["inner_bdry_function"],
                                 poly_order=0, filename_suffix="{}_{}".format(args.run_tag, kind),
                                 **kwargs)
        wt.euler_residual_conv_limit = args.residual_conv_limit
        wt.max_newton_iterations = args.rans_max_newton if kind == "rans" else args.euler_max_newton
        wt.log_cm_alpha = False
        wt.release_solver_memory = args.release
        wt.write_mesh_deformation = False
        if kind == "euler":
            wt.sim_model.fom_newton_inner_max_it = args.fom_newton_inner_max_it
        wt.set_up_sim()
        wt.postprocessor.log_cm_alpha = False

        shape_vector = self.shape.declare_design_variables()
        self.alpha_var = csdl.Variable(name='alpha', value=np.array([np.radians(2.0)]))
        self.mesh_motion_var = wt.deform_mesh()
        self.u_var = wt.evaluate(self.mesh_motion_var, shape_vector, self.alpha_var)
        recorder.stop()
        self.sim = csdl.experimental.JaxSimulator(
            recorder, gpu=False, additional_inputs=[self.alpha_var],
            additional_outputs=[self.u_var, self.mesh_motion_var])

        self.wt = wt
        self.model = wt.sim_model
        self.norms = StateNorms(self.model, wt.WALL_TAG)
        self.dvs = self.shape.dvs
        self.roms = {}
        self.continuation = None
        self.results = {v: DataStore() for v in args.variants}
        self.training_fom = DataStore()

    # -- shape parameters ------------------------------------------------------
    def shape_bounds(self):
        lower = np.concatenate([np.asarray(dv.lower, dtype=float).ravel() for dv in self.dvs.dvs])
        upper = np.concatenate([np.asarray(dv.upper, dtype=float).ravel() for dv in self.dvs.dvs])
        return lower, upper

    def set_shape(self, flat):
        offset = 0
        for name, dv in zip(self.dvs.names, self.dvs.dvs):
            n = int(np.prod(dv.shape))
            self.sim[self.dvs[name]] = np.asarray(flat[offset:offset + n]).reshape(dv.shape)
            offset += n

    # -- reduced-order models -------------------------------------------------
    def build_roms(self, args, parameter_scales):
        for variant in args.variants:
            rom = ReducedOrderModel(
                rb_size=args.rb_size, basis_mode=VARIANTS[variant], max_rank=args.max_rank,
                qr_rank_tol=args.qr_rank_tol, weightfunction=args.weightfunction,
                n_nonzero_weights=args.n_nonzero_weights, cubic_cutoff=args.cubic_cutoff,
                parameter_scales=parameter_scales, inner_product=args.inner_product,
                state_scaling=None if args.state_scaling == "none" else "freestream",
                newton_max_it=args.rom_newton_max_it, newton_rtol=args.rom_newton_rtol,
                newton_atol=args.rom_newton_atol, max_outer_iterations=args.rom_max_outer_iterations,
                g_conv_limit=args.rom_g_conv_limit, line_search=args.rom_line_search,
                min_snapshots=1)
            rom.set_up(self.model)
            self.roms[variant] = rom

    def add_snapshot(self, sample, solution_local):
        self.model.set_u_vec(solution_local)
        for rom in self.roms.values():
            rom.add_snapshot(self.model, sample)

    # -- solves ---------------------------------------------------------------
    def set_flow_condition(self, alpha, mach):
        self.model.set_mach(mach)
        self.wt.postprocessor.set_angle_of_attack(alpha)     # also sets the model's alpha

    def run_fom(self, sample, n_shape, args, label, initial_condition=None):
        """One FOM solve through the windtunnel's CSDL graph. Returns
        (solution_local, mesh_motions, walltime), or (None, None, walltime)
        for an invalid mesh or (without --keep-nonconverged) a failed solve.
        The mesh is at its baseline on return."""
        alpha, mach = float(sample[n_shape]), float(sample[n_shape + 1])
        if initial_condition is not None:
            self.wt.previous_solution = initial_condition
        elif not args.warm_start:
            self.wt.previous_solution = None
        self.set_flow_condition(alpha, mach)
        self.set_shape(sample[:n_shape])
        self.sim[self.alpha_var] = np.array([alpha])
        t0 = perf_counter()
        self.sim.run()
        walltime = perf_counter() - t0
        comm = self.model.mesh_obj.mesh.comm
        if not self.wt.mesh.is_valid:
            PETSc.Sys.Print("WARNING: [{} {}] inverted cells in the deformed mesh; sample "
                            "ignored".format(self.kind, label))
            return None, None, walltime
        solution = self.model.u_vec.x.petsc_vec.getArray().copy()
        converged = bool(self.model.last_solve_converged)
        finite = comm.allreduce(bool(np.all(np.isfinite(solution))), op=MPI.LAND)
        if not (converged and finite):
            PETSc.Sys.Print("WARNING: [{} {}] FOM solve {}; {}".format(
                self.kind, label, "did not converge" if finite else "is not finite",
                "kept (--keep-nonconverged)" if args.keep_nonconverged else "sample ignored"))
            if not args.keep_nonconverged or not finite:
                return None, None, walltime
        if not args.warm_start:
            self.wt.previous_solution = None
        return solution, np.array(self.sim[self.mesh_motion_var]), walltime

    def coefficients(self, solution_local):
        pp = self.wt.postprocessor
        L, cl = pp.compute_cl(solution_local)
        D, cd = pp.compute_cd(solution_local)
        M, cm = pp.compute_cm(solution_local)
        return dict(L=L, D=D, M=M, cl=cl, cd=cd, cm=cm)


# =============================================================================
# Sampling and Mach continuation
# =============================================================================
def lhs_sample(lower, upper, n, seed):
    """Latin-Hypercube samples; dimensions with lower == upper are held fixed."""
    free = upper > lower
    samples = np.tile(lower, (n, 1))
    if n > 0 and np.any(free):
        unit = qmc.LatinHypercube(d=int(np.sum(free)), seed=seed).random(n=n)
        samples[:, free] = qmc.scale(unit, lower[free], upper[free])
    return samples


def uniform_sample(lower, upper, n, rng):
    free = upper > lower
    samples = np.tile(lower, (n, 1))
    if n > 0 and np.any(free):
        samples[:, free] = rng.uniform(lower[free], upper[free], size=(n, int(np.sum(free))))
    return samples


class ContinuationStore:
    """
    Converged FOM solutions by parameter vector. select() returns the one at
    or below the requested Mach number that is closest in the parameter space
    normalized by the sampled ranges, the Mach component weighted by
    mach_weight. (Measured at p = 0 with indiscriminate warm starts: 48% of
    downward Mach jumps failed to converge, no upward one did.) Solutions are
    rank-local arrays.
    """

    def __init__(self, lower, upper, mach_weight=1.0):
        span = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
        self.span = np.where(np.abs(span) < 1e-300, 1.0, span)
        self.mach_weight = float(mach_weight)
        self.params, self.solutions = [], []

    def add(self, param_vec, solution_local):
        self.params.append(np.asarray(param_vec, dtype=float).copy())
        self.solutions.append(np.asarray(solution_local).copy())

    def select(self, param_vec, mach, mach_tol=1e-12):
        best, best_d, n_eligible = None, np.inf, 0
        q = np.asarray(param_vec, dtype=float)
        for j, p in enumerate(self.params):
            if p[-1] > mach + mach_tol:
                continue
            n_eligible += 1
            d = (q - p) / self.span
            d[-1] *= self.mach_weight
            dist = float(np.sqrt(d @ d))
            if dist < best_d:
                best, best_d = j, dist
        if best is None:
            return None, "free stream (no stored solution at Mach <= {:.4f})".format(mach)
        return self.solutions[best], "stored solution {} (Mach {:.4f} -> {:.4f}, distance {:.4f}, " \
                                     "{} eligible)".format(best, self.params[best][-1], mach, best_d,
                                                           n_eligible)


# =============================================================================
# Training data on disk (partition-independent: cells in input-mesh order)
# =============================================================================
def _owned_cell_blocks(model):
    V = model.functionspaces["V"]
    msh = V.mesh
    n_own = msh.topology.index_map(msh.topology.dim).size_local
    dofs = np.asarray(V.dofmap.list[:n_own]).ravel()      # one dof block per DG0 cell
    orig = np.asarray(msh.topology.original_cell_index[:n_own], dtype=np.int64)
    return orig, dofs, model.n_state


def gather_state(model, local):
    """A rank-local state array as (n_cells, n_state), input-mesh cell order, on rank 0."""
    orig, dofs, bs = _owned_cell_blocks(model)
    gathered = model.mesh_obj.mesh.comm.gather((orig, np.asarray(local).reshape(-1, bs)[dofs]), root=0)
    if gathered is None:
        return None
    n = sum(o.size for o, _ in gathered)
    out = np.empty((n, bs))
    for o, blocks in gathered:
        out[o] = blocks
    return out


def scatter_state(model, cells):
    """The rank-local state array of an (n_cells, n_state) input-order array."""
    orig, dofs, bs = _owned_cell_blocks(model)
    local = np.zeros((dofs.size, bs))
    local[dofs] = np.asarray(cells[orig])
    return local.ravel()


def training_paths(args, kind):
    stem = os.path.join(args.output_dir, "training_{}_{}_n{}_seed{}".format(
        args.dimension, kind, args.n_train, args.seed))
    return stem + "_meta.json", stem + "_snapshots.npy"


def write_training_data(args, problem, parameters, snapshots, lower, upper):
    comm = MPI.COMM_WORLD
    meta_path, snap_path = training_paths(args, problem.kind)
    if comm.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        np.save(snap_path, np.array(snapshots))
        with open(meta_path, "w") as f:
            json.dump({"mesh": os.path.abspath(problem.mesh_path), "model": problem.kind,
                       "n_cells": int(np.array(snapshots).shape[1]) if snapshots else 0,
                       "n_state": int(problem.model.n_state),
                       "parameters": np.asarray(parameters).tolist(),
                       "bounds_lower": np.asarray(lower).tolist(),
                       "bounds_upper": np.asarray(upper).tolist(),
                       "reynolds": getattr(problem, "reynolds", None)}, f)
        PETSc.Sys.Print("[{}] wrote {} training snapshots to {}".format(
            problem.kind, len(parameters), snap_path))
    comm.Barrier()


def read_training_data(args, problem):
    meta_path, snap_path = training_paths(args, problem.kind)
    with open(meta_path) as f:
        meta = json.load(f)
    n_cells = problem.model.mesh_obj.mesh.comm.allreduce(
        problem.model.mesh_obj.mesh.topology.index_map(problem.model.dimensions).size_local, op=MPI.SUM)
    if meta["mesh"] != os.path.abspath(problem.mesh_path) or meta["n_state"] != problem.model.n_state \
            or (meta["parameters"] and meta["n_cells"] != n_cells):
        raise ValueError("{} was written for another mesh or model ({}, {} cells, {} state "
                         "components)".format(meta_path, meta["mesh"], meta["n_cells"], meta["n_state"]))
    if problem.kind == "rans" and meta["reynolds"] != problem.reynolds:
        raise ValueError("{} was solved at Re {} (this run: {})".format(
            meta_path, meta["reynolds"], problem.reynolds))
    snapshots = np.load(snap_path, mmap_mode="r") if meta["parameters"] else []
    return np.array(meta["parameters"]), snapshots, np.array(meta["bounds_lower"]), \
        np.array(meta["bounds_upper"])


# =============================================================================
# Stages
# =============================================================================
def train(args, problems, lower, upper, n_shape):
    comm = MPI.COMM_WORLD
    if args.read_training_data:
        for pb in problems.values():
            parameters, snapshots, lo, up = read_training_data(args, pb)
            if not (np.allclose(lo, lower) and np.allclose(up, upper)):
                raise ValueError("the stored training data was sampled from other bounds")
            for sample, cells in zip(parameters, snapshots):
                local = scatter_state(pb.model, cells)
                pb.add_snapshot(sample, local)
                if pb.continuation is not None:
                    pb.continuation.add(sample, local)
            PETSc.Sys.Print("[{}] read {} training snapshots".format(pb.kind, len(parameters)))
        return

    samples = lhs_sample(lower, upper, args.n_train, args.seed)
    order = (sorted(range(args.n_train), key=lambda j: samples[j][-1])
             if args.mach_continuation and args.mach_continuation_sort else range(args.n_train))
    stored = {kind: ([], []) for kind in problems}
    for i in order:
        label = "training {}/{}".format(i + 1, args.n_train)
        PETSc.Sys.Print("[{}] alpha {:.4f} deg, Mach {:.4f}".format(
            label, np.degrees(samples[i][n_shape]), samples[i][-1]))
        for kind, pb in problems.items():
            ic = None
            if pb.continuation is not None:
                ic, why = pb.continuation.select(samples[i], samples[i][-1])
                PETSc.Sys.Print("[{} {}] initial condition: {}".format(kind, label, why))
            solution, motions, walltime = pb.run_fom(samples[i], n_shape, args, label, ic)
            if solution is None:
                continue
            if pb.continuation is not None:
                pb.continuation.add(samples[i], solution)
            pb.add_snapshot(samples[i], solution)
            pb.wt.mesh.apply_node_motions(motions)
            c = pb.coefficients(solution)
            pb.wt.mesh.reset_nodes()
            if comm.rank == 0:
                pb.training_fom.FOM_force_coefficients.append((i, c["cd"], c["cl"], c["cm"], np.nan))
                pb.training_fom.FOM_forces.append((i, c["D"], c["L"], c["M"], np.nan))
                pb.training_fom.FOM_walltime.append((i, walltime))
                pb.training_fom.parameter_vectors_per_iteration.append((i, samples[i]))
            cells = gather_state(pb.model, solution)
            if comm.rank == 0:
                stored[kind][0].append(samples[i])
                stored[kind][1].append(cells)
    for kind, pb in problems.items():
        write_training_data(args, pb, stored[kind][0], stored[kind][1], lower, upper)


def capture_phase_breakdown():
    """{phase: self seconds (max over ranks)} recorded since the last call, and
    a reset of the profiler, so each solve's breakdown starts empty; {} when
    profiling is off. Collective."""
    out = PROFILER.report(print_fn=lambda *args, **kwargs: None)
    PROFILER.reset()
    if not out:
        return {}
    return {name: float(t) for name, t in zip(out["names"], out["self_max"])}


def test(args, problems, lower, upper, n_shape):
    """The test sweep. Returns the phase breakdowns {kind: {"FOM" or variant:
    {test index: (phase breakdown, wall time)}}}."""
    comm = MPI.COMM_WORLD
    samples = uniform_sample(lower, upper, args.n_test, np.random.default_rng(args.seed + 1))
    order = (sorted(range(args.n_test), key=lambda j: samples[j][-1])
             if args.mach_continuation and args.mach_continuation_sort else range(args.n_test))
    # Phase timing for the test solves only (the training stage is a one-off
    # cost); one profiler is shared by both models, so it is drained after
    # every solve
    first = next(iter(problems.values()))
    first.model.enable_profiling(enabled=args.profiling, barriers=args.profile_barriers, reset=True)
    breakdowns = {kind: {name: {} for name in ["FOM"] + list(args.variants)} for kind in problems}
    for i in order:
        sample = samples[i]
        label = "test {}/{}".format(i + 1, args.n_test)
        PETSc.Sys.Print("[{}] alpha {:.4f} deg, Mach {:.4f}".format(
            label, np.degrees(sample[n_shape]), sample[-1]))
        for kind, pb in problems.items():
            ic = None
            if pb.continuation is not None:
                ic, why = pb.continuation.select(sample, sample[-1])
                PETSc.Sys.Print("[{} {}] initial condition: {}".format(kind, label, why))
            capture_phase_breakdown()                    # start the solve from an empty profile
            fom, motions, fom_walltime = pb.run_fom(sample, n_shape, args, label, ic)
            fom_breakdown = capture_phase_breakdown()
            if fom is None:
                continue
            if comm.rank == 0:
                breakdowns[kind]["FOM"][i] = (fom_breakdown, fom_walltime)
            if pb.continuation is not None:
                pb.continuation.add(sample, fom)
            pb.wt.mesh.apply_node_motions(motions)
            c_fom = pb.coefficients(fom)
            pb.wt.mesh.reset_nodes()
            PETSc.Sys.Print("[{} {}] FOM: c_l {:.6f}, c_d {:.6f}, c_m {:.6f} ({:.2f} s)".format(
                kind, label, c_fom["cl"], c_fom["cd"], c_fom["cm"], fom_walltime))
            for variant, rom in pb.roms.items():
                ds = pb.results[variant]
                if comm.rank == 0:
                    ds.FOM_force_coefficients.append((i, c_fom["cd"], c_fom["cl"], c_fom["cm"], np.nan))
                    ds.FOM_forces.append((i, c_fom["D"], c_fom["L"], c_fom["M"], np.nan))
                    ds.FOM_walltime.append((i, fom_walltime))
                    ds.parameter_vectors_per_iteration.append((i, sample))
                # Timed (and profiled): the ROM's own share of an evaluation --
                # deforming the mesh to this design (the warp is reused from
                # the FOM evaluation), the initial state and the reduced solve
                capture_phase_breakdown()
                t0 = perf_counter()
                pb.wt.mesh.apply_node_motions(motions)
                if args.rom_ic == "continuation" and ic is not None:
                    pb.model.set_u_vec(ic)
                else:
                    pb.model.interpolate_solution_vector(None)
                result = rom.solve(pb.model, sample)
                rom_walltime = perf_counter() - t0
                rom_breakdown = capture_phase_breakdown()
                if result is None or not np.all(np.isfinite(result.solution)):
                    PETSc.Sys.Print("[{} {} {}] no ROM solution".format(kind, label, variant))
                    if comm.rank == 0:
                        ds.rom_converged.append((i, False))
                    pb.wt.mesh.reset_nodes()
                    continue
                errors = pb.norms.errors(fom, result.solution)
                c_rom = pb.coefficients(result.solution)
                PETSc.Sys.Print(
                    "[{} {} {}] ROM: c_l {:.6f}, c_d {:.6f}, c_m {:.6f}; dc_l {:.3e}, dc_d {:.3e}, "
                    "dc_m {:.3e}; relative L2 error {:.3e}; eta {:.3e}; {} iterations, {:.2f} s".format(
                        kind, label, variant, c_rom["cl"], c_rom["cd"], c_rom["cm"],
                        c_rom["cl"] - c_fom["cl"], c_rom["cd"] - c_fom["cd"], c_rom["cm"] - c_fom["cm"],
                        errors[1], result.eta, result.iterations, rom_walltime))
                if comm.rank == 0:
                    breakdowns[kind][variant][i] = (rom_breakdown, rom_walltime)
                    ds.ROM_force_coefficients.append((i, c_rom["cd"], c_rom["cl"], c_rom["cm"], np.nan))
                    ds.ROM_forces.append((i, c_rom["D"], c_rom["L"], c_rom["M"], np.nan))
                    ds.ROM_walltime.append((i, rom_walltime))
                    ds.rom_relative_residuals.append((i, result.eta))
                    ds.rom_twonorm_residuals.append((i, result.residual_norm))
                    ds.rom_converged.append((i, bool(result.converged)))
                    ds.absolute_l2_errors_per_iteration.append((i, errors[0]))
                    ds.relative_l2_errors_per_iteration.append((i, errors[1]))
                    ds.absolute_l2_bdry_errors_per_iteration.append((i, errors[2]))
                    ds.relative_l2_bdry_errors_per_iteration.append((i, errors[3]))
                pb.wt.mesh.reset_nodes()
                if args.release:
                    pb.model.release_solver()
    first.model.enable_profiling(enabled=False)
    return breakdowns


# =============================================================================
# Summary and plots
# =============================================================================
def _paired(ds, col):
    """(ROM, FOM) values of tuple column `col`, paired by evaluation index."""
    rom = {v[0]: v[col] for v in ds.ROM_force_coefficients}
    fom = {v[0]: v[col] for v in ds.FOM_force_coefficients}
    shared = sorted(set(rom) & set(fom))
    return np.array([rom[i] for i in shared]), np.array([fom[i] for i in shared])


def metric_values(ds):
    """{metric: values over the test points} of one (model, variant)."""
    out = {"rel_l2": [v for _, v in ds.relative_l2_errors_per_iteration],
           "rel_l2_wall": [v for _, v in ds.relative_l2_bdry_errors_per_iteration],
           "eta": [v for _, v in ds.rom_relative_residuals],
           "rom_walltime": [v for _, v in ds.ROM_walltime],
           "fom_walltime": [v for _, v in ds.FOM_walltime]}
    for name, col in (("cd", 1), ("cl", 2), ("cm", 3)):
        rom, fom = _paired(ds, col)
        out["abs_d" + name] = np.abs(rom - fom)
        out["rel_d" + name] = np.abs(rom - fom) / np.abs(fom)
    return {k: np.asarray(v, dtype=float)[np.isfinite(np.asarray(v, dtype=float))]
            for k, v in out.items()}


def print_summary(args, problems):
    PETSc.Sys.Print("\n" + "=" * 78 + "\nPOD ROM benchmark summary ({} test points)".format(args.n_test))
    for kind, pb in problems.items():
        for variant in args.variants:
            ds = pb.results[variant]
            vals = metric_values(ds)
            PETSc.Sys.Print("--- {} {}: {} ROM solutions, {} converged ---".format(
                kind, variant, len(ds.ROM_walltime), sum(1 for _, ok in ds.rom_converged if ok)))
            for name in ("rel_l2", "rel_l2_wall", "abs_dcl", "abs_dcd", "abs_dcm", "eta",
                         "rom_walltime", "fom_walltime"):
                v = vals[name]
                if v.size == 0:
                    PETSc.Sys.Print("    {:13s} no values".format(name))
                    continue
                PETSc.Sys.Print("    {:13s} mean {:.3e}  p50 {:.3e}  p95 {:.3e}  max {:.3e}".format(
                    name, np.mean(v), np.percentile(v, 50), np.percentile(v, 95), np.max(v)))
            if vals["rom_walltime"].size and vals["fom_walltime"].size:
                PETSc.Sys.Print("    speed-up (mean FOM / mean ROM wall time): {:.2f}".format(
                    np.mean(vals["fom_walltime"]) / np.mean(vals["rom_walltime"])))
    PETSc.Sys.Print("=" * 78)


# Reference palette of the dataviz method: categorical slots 1-2 (validated
# all-pairs for up to three slots), light chart surface and text inks
SERIES = ("#2a78d6", "#eb6834")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def plot_metric(args, problems, metric, ylabel, path, fom_reference=False):
    """Small multiples, one panel per flow model on a shared log axis: the
    distribution over the test points of `metric`, one box per variant."""
    kinds = list(problems)
    fig, axes = plt.subplots(1, len(kinds), figsize=(1.0 + 2.6 * len(kinds), 3.4), sharey=True,
                             squeeze=False, facecolor=SURFACE)
    for ax, kind in zip(axes[0], kinds):
        ax.set_facecolor(SURFACE)
        data = [metric_values(problems[kind].results[v])[metric] for v in args.variants]
        positions = np.arange(1, len(args.variants) + 1)
        for pos, values, color in zip(positions, data, SERIES):
            if values.size == 0 or not np.all(values > 0):
                ax.text(pos, 0.5, "no data", transform=ax.get_xaxis_transform(), ha="center",
                        color=INK_2, fontsize=8)
                continue
            ax.boxplot(values, positions=[pos], widths=0.55, patch_artist=True,
                       boxprops=dict(facecolor=color + "40", edgecolor=color, linewidth=1.5),
                       medianprops=dict(color=INK, linewidth=1.5),
                       whiskerprops=dict(color=color, linewidth=1.2),
                       capprops=dict(color=color, linewidth=1.2),
                       flierprops=dict(marker="o", markersize=5, markerfacecolor=color,
                                       markeredgecolor=SURFACE, markeredgewidth=1.0))
        if fom_reference:
            fom = metric_values(problems[kind].results[args.variants[0]])["fom_walltime"]
            if fom.size:
                ax.axhline(np.mean(fom), color=INK_2, linestyle="--", linewidth=1.2)
                ax.text(positions[-1] + 0.45, np.mean(fom), "FOM mean", color=INK_2, fontsize=8,
                        ha="right", va="bottom")
        ax.set_yscale("log")
        ax.set_xticks(positions, labels=args.variants)
        ax.set_xlim(0.4, len(args.variants) + 0.6)
        ax.set_title(kind.upper() if kind == "rans" else kind.capitalize(), color=INK, fontsize=10)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(INK_2)
        ax.tick_params(which="both", colors=INK_2, labelsize=8)
    axes[0][0].set_ylabel(ylabel, color=INK, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c + "40", edgecolor=c, linewidth=1.5)
               for c in SERIES[:len(args.variants)]]
    fig.legend(handles, args.variants, loc="upper center", ncol=len(args.variants), frameon=False,
               fontsize=8, labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# Phase groups (utils/solver_profiling.py) in plotting order; "other" is the
# wall time no phase covers (mainly the CSDL graph dispatch). Colors: the
# reference categorical palette in its fixed slot order (validated for adjacent
# pairs, as in a stacked bar), "other" neutral gray as overhead.
PHASE_GROUPS = ["assemble", "solve", "rom", "limiter", "setup", "mesh", "output", "post"]
OTHER = "other"
PHASE_COLORS = dict(zip(PHASE_GROUPS, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                       "#008300", "#4a3aa7", "#e34948")))
PHASE_COLORS[OTHER] = "#9a9890"


def mean_group_breakdown(records):
    """{index: (phase breakdown, wall time)} -> ({group: mean seconds}, mean
    wall time); "other" is the wall time minus the phases at each point, so the
    groups add up to the wall time."""
    if not records:
        return {}, 0.0
    sums, total = {}, 0.0
    for breakdown, walltime in records.values():
        grouped = {}
        for name, t in breakdown.items():
            g = default_group(name)
            grouped[g] = grouped.get(g, 0.0) + t
        grouped[OTHER] = max(0.0, walltime - sum(grouped.values()))
        for g, t in grouped.items():
            sums[g] = sums.get(g, 0.0) + t
        total += walltime
    n = len(records)
    return {g: t / n for g, t in sums.items()}, total / n


def ordered_groups(present):
    out = [g for g in PHASE_GROUPS if g in present]
    out += sorted(g for g in present if g not in PHASE_GROUPS and g != OTHER)
    return out + ([OTHER] if OTHER in present else [])


def print_phase_summary(args, breakdowns):
    PETSc.Sys.Print("\nMean wall time per solve by phase group [s] (share)")
    for kind, by_model in breakdowns.items():
        for name in ["FOM"] + list(args.variants):
            means, total = mean_group_breakdown(by_model[name])
            if total == 0.0:
                continue
            parts = ", ".join("{} {:.3g} ({:.0%})".format(g, means[g], means[g] / total)
                              for g in ordered_groups(set(means)) if means[g] > 0)
            PETSc.Sys.Print("  {} {}: {:.3g} s -- {}".format(kind, name, total, parts))
            if name == "FOM" and means.get("assemble", 0.0) > 0:
                PETSc.Sys.Print("    assembly fraction {:.1%}: a projection-only ROM can be at most "
                                "{:.1f}x faster per Newton step".format(
                                    means["assemble"] / total, total / means["assemble"]))


def plot_phase_breakdown(args, problems, breakdowns, plot_dir):
    """Stacked bars of the mean wall time per solve by phase group, in seconds
    and as a share of the solve: one panel per flow model, one bar for the
    FOM and one per variant."""
    kinds = list(problems)
    labels = ["FOM"] + list(args.variants)
    means = {kind: [mean_group_breakdown(breakdowns[kind][n]) for n in labels] for kind in kinds}
    groups = ordered_groups({g for kind in kinds for m, _ in means[kind] for g in m})
    if not groups:
        PETSc.Sys.Print("No phase breakdown recorded; skipping the wall-time breakdown plots.")
        return
    for percent, ylabel, name in ((False, "Mean wall time per solve [s]", "walltime_breakdown_seconds"),
                                  (True, "Share of the solve wall time [%]", "walltime_breakdown_percent")):
        fig, axes = plt.subplots(1, len(kinds), figsize=(1.2 + 2.8 * len(kinds), 3.8), squeeze=False,
                                 sharey=percent, facecolor=SURFACE)
        for ax, kind in zip(axes[0], kinds):
            ax.set_facecolor(SURFACE)
            x = np.arange(len(labels))
            bottom = np.zeros(len(labels))
            totals = np.array([t for _, t in means[kind]])
            for g in groups:
                h = np.array([m.get(g, 0.0) for m, _ in means[kind]])
                if percent:
                    h = np.where(totals > 0, 100.0 * h / np.where(totals > 0, totals, 1.0), 0.0)
                ax.bar(x, h, bottom=bottom, width=0.6, color=PHASE_COLORS.get(g, PHASE_COLORS[OTHER]),
                       edgecolor=SURFACE, linewidth=1.0, label=g)
                bottom += h
            ax.set_xticks(x, labels=labels)
            ax.set_title(kind.upper() if kind == "rans" else kind.capitalize(), color=INK, fontsize=10)
            ax.yaxis.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(INK_2)
            ax.tick_params(which="both", colors=INK_2, labelsize=8)
        axes[0][0].set_ylabel(ylabel, color=INK, fontsize=9)
        handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS.get(g, PHASE_COLORS[OTHER])) for g in groups]
        fig.legend(handles, groups, loc="lower center", ncol=min(len(groups), 5), frameon=False,
                   fontsize=8, labelcolor=INK_2)
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(os.path.join(plot_dir, name + ".png"), dpi=150, facecolor=SURFACE)
        plt.close(fig)


def write_results(args, problems, test_results=True, breakdowns=None):
    comm = MPI.COMM_WORLD
    if comm.rank != 0:
        return
    out_dir = os.path.join(args.output_dir, "Result_dicts")
    os.makedirs(out_dir, exist_ok=True)
    for kind, pb in problems.items():
        n_dofs = pb.model.u_vec.x.petsc_vec.getSize()
        if test_results:
            for variant, ds in pb.results.items():
                ds.fe_dofs, ds.num_var = n_dofs, int(pb.dvs.size + 2)
                ds.write_store_to_numpy_file(out_dir, "{}_{}_{}.npy".format(args.run_tag, kind, variant))
        pb.training_fom.fe_dofs = n_dofs
        if pb.training_fom.FOM_walltime:
            pb.training_fom.write_store_to_numpy_file(out_dir, "{}_{}_training_fom.npy".format(
                args.run_tag, kind))
    if not test_results:
        PETSc.Sys.Print("Wrote the training FOM records to {}/".format(out_dir))
        return
    plot_dir = os.path.join(args.output_dir, "plots_{}".format(args.run_tag))
    os.makedirs(plot_dir, exist_ok=True)
    for metric, ylabel, name, ref in (
            ("rel_l2", "Relative state L2 error", "l2_relative_error", False),
            ("rel_l2_wall", "Relative wall state L2 error", "l2_wall_relative_error", False),
            ("abs_dcl", r"Absolute $c_l$ error", "cl_absolute_error", False),
            ("abs_dcd", r"Absolute $c_d$ error", "cd_absolute_error", False),
            ("eta", r"ROM residual $\|R\|/\|u\|$", "rom_eta", False),
            ("rom_walltime", "ROM wall time [s]", "walltime", True)):
        plot_metric(args, problems, metric, ylabel, os.path.join(plot_dir, name + ".png"),
                    fom_reference=ref)
    if breakdowns:
        np.save(os.path.join(args.output_dir, "walltime_breakdown_{}.npy".format(args.run_tag)),
                {"variants": list(args.variants), "breakdowns": breakdowns})
        plot_phase_breakdown(args, problems, breakdowns, plot_dir)
    PETSc.Sys.Print("Wrote DataStores to {}/ and plots to {}/".format(out_dir, plot_dir))


# =============================================================================
def main(argv=None):
    args = parse_args(argv)
    PETSc.Sys.Print("benchmark_pod: {}".format(json.dumps(vars(args), default=str)))
    definition = problem_definition(args.dimension)
    problems = {kind: Problem(kind, args, definition) for kind in args.models}

    # The same design points for every model: the shape parameterizations
    # must agree in size and bounds
    first = next(iter(problems.values()))
    shape_lower, shape_upper = first.shape_bounds()
    for pb in problems.values():
        lo, up = pb.shape_bounds()
        if lo.shape != shape_lower.shape or not (np.allclose(lo, shape_lower) and np.allclose(up, shape_upper)):
            raise ValueError("the {} and {} shape parameterizations differ".format(first.kind, pb.kind))
    n_shape = shape_lower.size
    lower = np.concatenate([args.cp_motion_sample_scale * shape_lower,
                            [np.radians(args.alpha_lower), args.mach_lower]])
    upper = np.concatenate([args.cp_motion_sample_scale * shape_upper,
                            [np.radians(args.alpha_upper), args.mach_upper]])
    span = upper - lower
    parameter_scales = (np.where(span > 0, 1.0 / np.where(span > 0, span, 1.0), 0.0)
                        if args.parameter_distance == "normalized" else None)
    for pb in problems.values():
        pb.build_roms(args, parameter_scales)
        if args.mach_continuation:
            pb.continuation = ContinuationStore(lower, upper, args.mach_distance_weight)
    PETSc.Sys.Print("Design space: {} shape parameters, alpha [{}, {}] deg, Mach [{}, {}]".format(
        n_shape, args.alpha_lower, args.alpha_upper, args.mach_lower, args.mach_upper))

    train(args, problems, lower, upper, n_shape)
    for pb in problems.values():
        for variant, rom in pb.roms.items():
            PETSc.Sys.Print("[{} {}] {} snapshots, stored rank {}".format(
                pb.kind, variant, rom.n_snapshots, rom.pod.snapshot_matrix.rank))
    if args.training_only:
        write_results(args, problems, test_results=False)
        PETSc.Sys.Print("Training stage only (--training-only): no test points evaluated.")
        return
    breakdowns = test(args, problems, lower, upper, n_shape)
    print_summary(args, problems)
    if args.profiling:
        print_phase_summary(args, breakdowns)
    write_results(args, problems, breakdowns=breakdowns if args.profiling else None)


if __name__ == '__main__':
    main()
