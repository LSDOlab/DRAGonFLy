"""
Mesh sensitivity study of ONERA's OAT15A airfoil: steady SA-neg RANS at p = 0
on structured grid levels, at the conditions of DPW8 test case 1a: M 0.73,
Re 3e6 (chord), static temperature 271 K, alpha 1.36, 1.50, 2.50, 3.00,
3.10 deg. Grid families (--mesh-family): the Cadence REV02 grids of DPW8 test
case 1c, levels 1-6, far field at 1000 chords (cadence_rev02, the default;
meshes/ONERA-OAT15A_Cadence_Structured_REV02/), the Cadence REV01 grids,
levels 1-6, far field at 150 chords (cadence;
meshes/Cadence-ONERA-OAT15A_..._Structured/), or the Rizzi grids 1-7 (rizzi;
meshes/ONERA-ONERA-OAT15A-Rizzi/).

The grids are structured multi-block CGNS files one cell thick in span (chord
230 mm); utils/mesh_io_utils reduces each to a 2D quad mesh with chord 1 in
memory. The cases run one at a time. The default model is SA-neg RANS with MUSCL,
and each case starts from the converged first-order RANS solution (the same model
without MUSCL), which itself starts from the free stream (--init first-order).
--init freestream starts from the free stream directly; --init euler from the
first-order Euler solution. --model euler runs the Euler equations (slip wall):
first-order (HLL flux) by default, from the free stream; --muscl adds MUSCL
reconstruction, with the Van Albada limiter (epsilon 0.1) as its default (--limiter,
--limiter-eps). A case that does not converge is recorded as such and the sweep
goes on.

    OMP_NUM_THREADS=1 mpirun -n 8 python oat15a_analysis.py
    OMP_NUM_THREADS=1 mpirun -n 8 python oat15a_analysis.py --grids 1 2 --alphas 2.5
    python oat15a_analysis.py --plot <run folder>/results.csv --figure-dir figures

Every run writes into its own new folder under --out-dir:

  results.csv               one row per case, rewritten after every case:
                            c_l, c_d, its pressure and friction parts
                            (c_d_pressure + c_d_friction = c_d), c_m, the
                            solve's wall time, the dof count, convergence
  wall_gridN_aA.csv         C_p and c_f at the ends and middle of every wall
                            facet
  solution_gridN_aA.npz     the converged state (utils/checkpoint.save_checkpoint,
                            partition independent) and its case settings; rebuild
                            the model on the same grid and fill it with
                            load_checkpoint to post-process, e.g. field C_p
  config.json, figures      the settings, and the figures of plot_results
  grid*_a*_{solution,pressure,mach}.bp   VTX fields, with --write-fields

The model set-up, force coefficients and wall output are those of
../flow_analysis.py.
"""
import argparse
import gc
import json
import os
import sys
import tempfile
import time

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

from dragonfly_sim.utils.mesh_io_utils import load_dolfinx_mesh
from dragonfly_sim.utils.checkpoint import save_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
from flow_analysis import WALL_TAG, build_model, ForceCoefficients, wall_writer, field_writer  # noqa: E402

MESH_DIR = os.path.join(HERE, os.pardir, os.pardir, "meshes")
# grid families: (file pattern for level N, levels available)
_CADENCE = "Cadence-ONERA-OAT15A_230mmChord_780mmSpan_upZ_2024_09_05_Structured"
GRID_FILES = {
    "cadence": (os.path.join(_CADENCE, "ONERA-OAT15A_230mmChord_780mmSpan_upZ_2024_09_05_Structured_Level-{0}.cgns"),
                [1, 2, 3, 4, 5, 6]),
    "rizzi": (os.path.join("ONERA-ONERA-OAT15A-Rizzi", "grid{0}", "OAT15A_Rizzi_{0}.cgns"), [1, 2, 3, 4, 5, 6, 7]),
    # DPW8 test case 1c grids: far field at 1000 chords (REV01: 150)
    "cadence_rev02": (os.path.join("ONERA-OAT15A_Cadence_Structured_REV02", "ONERA-OAT15A_Cadence_Structured_REV02",
                                   "CGNS", "ONERA-OAT15A_Cadence_Structured_REV02_Level-{0}.cgns"), [1, 2, 3, 4, 5, 6]),
}
COLUMNS = ("grid", "cells", "dofs", "alpha", "converged", "steps", "res", "wall_s", "case_s", "init_steps", "init_s",
           "c_l", "c_d", "c_d_pressure", "c_d_friction", "c_m")
FORMATS = {"grid": "%d", "cells": "%d", "dofs": "%d", "alpha": "%.4f", "converged": "%d", "steps": "%d", "init_steps": "%d", "init_s": "%.1f",
           "res": "%.6e", "wall_s": "%.1f", "case_s": "%.1f"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh-family", choices=sorted(GRID_FILES), default="cadence_rev02",
                   help="structured CGNS grid family: Cadence REV02 levels 1-6 (default), Cadence REV01 "
                        "levels 1-6, or Rizzi grids 1-7")
    p.add_argument("--grids", type=int, nargs="+", default=None,
                   help="grid levels (default: all levels of the family)")
    p.add_argument("--alphas", type=float, nargs="+", default=[1.36, 1.50, 2.50, 3.00, 3.10],
                   help="angles of attack [deg]")
    p.add_argument("--mach", type=float, default=0.73)
    p.add_argument("--Re", type=float, default=3e6, help="Reynolds number based on the chord")
    p.add_argument("--T-inf", type=float, default=271.0,
                   help="static free-stream temperature [K] (Sutherland); DPW8 test case 1a: 271 K")
    p.add_argument("--mesh-dir", default=MESH_DIR)
    # steady solver
    p.add_argument("--cfl0", type=float, default=5.0)
    p.add_argument("--cfl-rise-min", type=float, default=0.7,
                   help="PTC: the largest CFL cut on a residual rise (CFL *= residual ratio, not below this); "
                        "the controller's own default is 0.1")
    p.add_argument("--cfl-rise-max", type=float, default=0.9,
                   help="PTC: the mildest CFL cut once the residual rises by more than 5%% (the controller's "
                        "own default is 0.5, i.e. at least halved)")
    p.add_argument("--krylov-rtol", type=float, default=1e-3,
                   help="relative tolerance of the Krylov solve of every Newton (pseudo-time) step, at most 300 "
                        "iterations")
    p.add_argument("--cfl-max", type=float, default=None,
                   help="upper bound of the PTC CFL number (default: the controller's, 1e12)")
    p.add_argument("--steady-tol", type=float, default=1e-9)
    p.add_argument("--max-steps", type=int, default=1000, help="PTC step budget per case")
    p.add_argument("--model", choices=("sa", "euler"), default="sa",
                   help="flow model: SA-neg RANS with MUSCL (default), or the Euler equations (slip wall), "
                        "first-order unless --muscl is given")
    p.add_argument("--muscl", action="store_true",
                   help="--model euler: add MUSCL reconstruction (the RANS model always has it)")
    p.add_argument("--init", choices=("first-order", "freestream", "euler", "rans1"), default="first-order",
                   help="initial state of the MUSCL solve: the default is the converged first-order RANS solution "
                        "for --model sa and the free stream for --model euler; explicitly: the free stream, the "
                        "first-order Euler (HLL flux, no MUSCL, slip wall) or the first-order RANS solution. The "
                        "first-order solve is on the same mesh and angle of attack")
    p.add_argument("--init-tol", type=float, default=1e-8, help="residual of the first-order initialization")
    p.add_argument("--init-max-steps", type=int, default=600, help="PTC step budget of the first-order initialization")
    p.add_argument("--limiter", choices=("none", "vanalbada"), default=None,
                   help="slope limiter of the MUSCL reconstruction (default: none for --model sa, vanalbada for "
                        "--model euler --muscl, whose unlimited form froze at the trailing edge and the shock)")
    p.add_argument("--limiter-eps", type=float, default=None,
                   help="Van Albada smoothness parameter (default: 0.1 for --model euler --muscl, 0.01 for "
                        "--model sa)")
    p.add_argument("--report-limiter", action="store_true",
                   help="print which cells block the Newton step whenever the positivity limiter cuts it")
    # output
    p.add_argument("--fluid", choices=("air", "water"), default="air",
                   help="water: constant density at --water-speed and T_inf, preconditioned (Re as given)")
    p.add_argument("--water-speed", type=float, default=5.0, help="free-stream speed of water [m/s]")
    p.add_argument("--preconditioning", action="store_true",
                   help="local low-Mach (Weiss-Smith) preconditioning of the flux and pseudo-time term")
    p.add_argument("--precond-floor", type=float, default=1.0,
                   help="reference velocity floor of the preconditioning, in free-stream speeds")
    p.add_argument("--av", action="store_true",
                   help="Barter-Darmofal artificial viscosity (core/shock_capturing.py)")
    p.add_argument("--av-C", type=float, default=0.5, help="artificial viscosity coefficient")
    p.add_argument("--av-eta", type=float, default=0.1, help="artificial viscosity sensor scale")
    p.add_argument("--av-no-ducros", action="store_true", help="artificial viscosity without the Ducros switch")
    p.add_argument("--av-directional", action="store_true",
                   help="artificial viscosity only on the facets crossing the pressure gradient")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--label", default="")
    p.add_argument("--write-fields", action="store_true",
                   help="also VTX solution, pressure and Mach fields per case, for ParaView")
    p.add_argument("--plot", default=None, metavar="RESULTS_CSV", help="only plot an existing results.csv")
    p.add_argument("--figure-dir", default=None,
                   help="where --plot writes the figures (default: next to the CSV)")
    p.add_argument("--format", default="svg", help="figure file format (svg renders in a README)")
    return p.parse_args()


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
# Grid level is ordinal (coarse -> fine): one blue ramp, light -> dark, with a
# marker per grid so identity never rests on colour alone. Angles of attack are
# categories: the fixed categorical order.
GRID_COLOURS = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#1c5cab", "#104281", "#0d366b"]
GRID_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
ALPHA_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"alpha": r"angle of attack $\alpha$ [deg]", "c_l": r"lift coefficient $c_l$",
          "c_d": r"drag coefficient $c_d$", "c_m": r"moment coefficient $c_m$ (quarter chord)",
          "dofs": "degrees of freedom", "wall_s": "wall time of the solve [s]"}


def _figure(plt):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), facecolor=SURFACE, constrained_layout=True)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def _series(ax, x, y, ok, colour, marker, label):
    """A line through all points; filled markers where converged, hollow where not."""
    ax.plot(x, y, color=colour, linewidth=2, zorder=2, label=label)
    ax.plot(x[ok], y[ok], linestyle="none", marker=marker, markersize=7, color=colour,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.plot(x[~ok], y[~ok], linestyle="none", marker=marker, markersize=7, markerfacecolor=SURFACE,
            markeredgecolor=colour, markeredgewidth=1.5, zorder=3)


def _legend(fig, ax, colours, markers, labels, any_failed):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, linewidth=2, marker=m, markersize=7, markeredgecolor=SURFACE)
               for c, m in zip(colours, markers)]
    labels = list(labels)
    if any_failed:
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor=SURFACE,
                              markeredgecolor=INK_MUTED, markeredgewidth=1.5))
        labels.append("not converged")
    fig.legend(handles, labels, loc="outside right center", frameon=False)


def _plain_number(v, _pos=None):
    if v >= 1e6:
        return "{:g}M".format(v / 1e6)
    if v >= 1e3:
        return "{:g}k".format(v / 1e3)
    return "{:g}".format(v)


def plot_results(csv_path, figure_dir, fmt="svg"):
    """
    The mesh sensitivity figures from a results.csv, written to figure_dir:

      cl_alpha  c_l against alpha        one line per grid (its alpha sweep)
      cl_cd     c_l against c_d          one line per grid
      cm_cl     c_m against c_l          one line per grid
      dofs_wall wall time of the solve against the grid's dof count, one line
                per angle of attack (log-log)

    Hollow markers are cases that did not converge. Returns the file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
                         "svg.fonttype": "path"})
    data = np.atleast_1d(np.genfromtxt(csv_path, delimiter=",", names=True))
    any_failed = bool(np.any(data["converged"] <= 0))
    grids = sorted({int(g) for g in data["grid"]})
    os.makedirs(figure_dir, exist_ok=True)
    paths = []

    def save(fig, name):
        path = os.path.join(figure_dir, "{}.{}".format(name, fmt))
        fig.savefig(path, facecolor=SURFACE, dpi=200)
        plt.close(fig)
        paths.append(path)

    # per-grid alpha sweeps
    for name, x, y in (("cl_alpha", "alpha", "c_l"), ("cl_cd", "c_d", "c_l"), ("cm_cl", "c_l", "c_m")):
        fig, ax = _figure(plt)
        colours, markers, labels = [], [], []
        for g in grids:
            rows = np.sort(data[data["grid"] == g], order="alpha")
            k = g - 1
            colour, marker = GRID_COLOURS[k % len(GRID_COLOURS)], GRID_MARKERS[k % len(GRID_MARKERS)]
            label = "grid {} ({:,} cells)".format(g, int(rows["cells"][0]))
            _series(ax, rows[x], rows[y], rows["converged"] > 0, colour, marker, label)
            colours.append(colour), markers.append(marker), labels.append(label)
        ax.set_xlabel(LABELS[x])
        ax.set_ylabel(LABELS[y])
        _legend(fig, ax, colours, markers, labels, any_failed)
        save(fig, name)

    # cost: wall time against dofs, one line per angle of attack
    fig, ax = _figure(plt)
    alphas = sorted({round(float(a), 4) for a in data["alpha"]})
    colours, labels = [], []
    for k, a in enumerate(alphas):
        rows = np.sort(data[np.isclose(data["alpha"], a)], order="dofs")
        colour = ALPHA_COLOURS[k % len(ALPHA_COLOURS)]
        label = "\u03b1 = {:g}\u00b0".format(a)
        _series(ax, rows["dofs"], rows["wall_s"], rows["converged"] > 0, colour, "o", label)
        colours.append(colour), labels.append(label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        # 1-2-5 ticks with plain labels (100k, 200k; 20, 50, 100 s)
        axis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(ticker.FuncFormatter(_plain_number))
        axis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel(LABELS["dofs"])
    ax.set_ylabel(LABELS["wall_s"])
    _legend(fig, ax, colours, ["o"] * len(colours), labels, any_failed)
    save(fig, "dofs_wall")
    return paths


# ----------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------
def count_wall_facets(mesh_obj):
    """Global number of facets tagged as wall."""
    msh = mesh_obj.mesh
    facets = mesh_obj.meshtags.find(WALL_TAG)
    n_owned = np.sum(facets < msh.topology.index_map(msh.topology.dim - 1).size_local)
    return msh.comm.allreduce(int(n_owned), MPI.SUM)


def _ptc_settings(args):
    return {"cfl0": args.cfl0, "rise_factor_min": args.cfl_rise_min, "rise_factor_max": args.cfl_rise_max,
            **({} if args.cfl_max is None else {"cfl_max": args.cfl_max})}


def euler_muscl(args):
    return getattr(args, "model", "sa") == "euler" and bool(getattr(args, "muscl", False))


def resolve_limiter(args):
    """(limiter or None, Van Albada eps) of the MUSCL solve: Van Albada with eps 0.1 for the Euler model with
    MUSCL, no limiter for the RANS model (eps 0.01 if one is asked for)."""
    euler = euler_muscl(args)
    limiter = getattr(args, "limiter", None) or ("vanalbada" if euler else "none")
    if getattr(args, "model", "sa") == "euler" and not euler and limiter != "none":
        raise ValueError("--limiter needs MUSCL: add --muscl, or use --model sa")
    eps = getattr(args, "limiter_eps", None)
    return (None if limiter == "none" else limiter), (eps if eps is not None else (0.1 if euler else 1e-2))


def resolve_init(args):
    """The starting state --init names: "first-order" is the first-order RANS solution for the RANS model;
    the Euler model (already first-order, or with MUSCL) starts from the free stream unless the first-order Euler
    solution is asked for (--init euler)."""
    model = getattr(args, "model", "sa")
    init = getattr(args, "init", "first-order")
    if init == "first-order":
        init = "rans1" if model == "sa" else "freestream"
    if init == "rans1" and model == "euler":
        raise ValueError("--init rans1 needs --model sa: the Euler state has no turbulence variable")
    return init


def set_up_solver(model, args):
    """model.set_up_solver(); an Euler model with MUSCL needs the RANS model's Newton Jacobian wiring (the exact
    Jacobian for the Newton matrix, its compact part as the preconditioner) done by hand."""
    model.set_up_solver()
    if euler_muscl(args) and not model.direct_linear_solver:
        model.problem.configure_snes(model.solver.snes, "compact_pc")
        model.solver.krylov_solver.setFromOptions()


def first_order_state(msh, args, alpha, kind):
    """
    Solve a first-order model on `msh` at the case's Mach number, angle of
    attack and far field, to steady state by PTC: kind "euler" is the Euler
    equations (HLL flux, no MUSCL, slip wall); "rans1" is the RANS model
    without MUSCL (first-order HLL inviscid flux, no-slip wall, SA-neg).
    Returns (state, cell-to-block map, info dict): the state as an
    (n_blocks, n_state) array in that model's own block order, with the
    dofmap that maps each cell to its block, so that a model on the same mesh
    can be filled cell by cell. The model is freed before returning.
    """
    if kind == "euler":
        mesh_obj, model, _ = build_model(msh, model="euler", mach=args.mach, alpha_deg=alpha,
                                         **precond_options(args))
    else:
        mesh_obj, model, _ = build_model(msh, model="sa", mach=args.mach, alpha_deg=alpha, Re=args.Re,
                                         T_inf=args.T_inf, muscl=False, **precond_options(args))
    model.report_positivity_limiter = args.report_limiter
    model.krylov_rtol = args.krylov_rtol
    model.use_ptc = True
    model.ptc_settings = {**model.ptc_settings, **_ptc_settings(args)}
    model.max_newton_iterations = args.init_max_steps
    model.residual_conv_limit = args.init_tol
    model.set_up_solver()
    tic = time.perf_counter()
    r = model.solve_system().norm()
    info = dict(steps=model.time_integrator.last_steady_steps, res=r, wall_s=time.perf_counter() - tic,
                converged=bool(model.solver_converged) and np.isfinite(r))
    cl, cd, _, cm = ForceCoefficients(model, mesh_obj, alpha, aero_centre=(0.25, 0.0))()
    info.update(c_l=cl, c_d=cd, c_m=cm)
    bs = model.n_state
    state = model.u_vec.x.array.reshape(-1, bs).copy()
    cell_block = model.functionspaces["V"].dofmap.list[:, 0].copy()
    model.solver.destroy()
    del model, mesh_obj
    gc.collect()
    PETSc.garbage_cleanup(msh.comm)
    return state, cell_block, info


def precond_options(args):
    """build_model's low-Mach preconditioning keywords."""
    return dict(preconditioning=args.preconditioning or None, preconditioning_floor=args.precond_floor,
                fluid=args.fluid, water_speed=args.water_speed)


def av_options(args):
    """build_model's shock_capturing option."""
    if not args.av:
        return None
    return dict(C=args.av_C, eta=args.av_eta, ducros=False if args.av_no_ducros else None,
                directional=args.av_directional)


def apply_state(model, state, cell_block):
    """Overwrite the first state.shape[1] components of `model`'s state with
    `state` (see first_order_state), cell by cell: the mean flow of an Euler
    solution (the transported scalars keep their free-stream values), or the
    whole state of a first-order RANS solution."""
    bs = model.n_state
    u = model.u_vec.x.array.reshape(-1, bs)
    u[model.functionspaces["V"].dofmap.list[:, 0], :state.shape[1]] = state[cell_block]
    model.u_vec.x.scatter_forward()


def solve_case(msh, args, alpha, n_wall_edges):
    """Build the model on `msh`, check that its wall is the grid's
    `n_wall_edges` Wall edges, and solve to steady state from the converged
    first-order RANS solution (--init first-order, the default for --model sa;
    --init euler / rans1 name the first-order model; --init freestream, the
    default for --model euler: the free stream). Returns (model, mesh_obj,
    record dict)."""
    tic_case = time.perf_counter()
    init = dict(steps=0, wall_s=0.0)
    euler = None
    init_kind = resolve_init(args)
    if init_kind != "freestream":
        euler = first_order_state(msh, args, alpha, init_kind)
        init = euler[2]
        PETSc.Sys.Print("first-order {} start: converged={} steps={} |R|={:.3e} c_l={:.4f} c_d={:.5f} "
                        "c_m={:.4f} [{:.1f} s]".format(init_kind, init["converged"], init["steps"], init["res"],
                                                        init["c_l"], init["c_d"], init["c_m"], init["wall_s"]))
        if not init["converged"]:
            PETSc.Sys.Print("WARNING: the first-order initialization did not converge; the MUSCL solve starts "
                            "from its last state anyway")
    limiter, limiter_eps = resolve_limiter(args)
    mesh_obj, model, _ = build_model(msh, model=args.model, mach=args.mach, alpha_deg=alpha, Re=args.Re,
                                     T_inf=args.T_inf, limiter=limiter,
                                     limiter_eps=limiter_eps, euler_muscl=euler_muscl(args),
                                     shock_capturing=av_options(args), **precond_options(args))
    PETSc.Sys.Print("model {}{}: MUSCL limiter {} (eps {})".format(
        args.model, " with MUSCL" if args.model == "euler" and euler_muscl(args) else "",
        limiter or "none", limiter_eps if limiter else "-"))
    n_wall = count_wall_facets(mesh_obj)
    if n_wall != n_wall_edges:
        raise RuntimeError("the solver's wall has {} facets, the CGNS Wall family {} edges"
                           .format(n_wall, n_wall_edges))
    model.report_positivity_limiter = args.report_limiter
    model.krylov_rtol = args.krylov_rtol
    model.use_ptc = True
    model.ptc_settings = {**model.ptc_settings, **_ptc_settings(args)}
    model.max_newton_iterations = args.max_steps
    model.residual_conv_limit = args.steady_tol
    set_up_solver(model, args)
    if euler is not None:
        apply_state(model, euler[0], euler[1])
    tic = time.perf_counter()
    r = model.solve_system().norm()
    wall = time.perf_counter() - tic
    cl, cd, cdf, cm = ForceCoefficients(model, mesh_obj, alpha, aero_centre=(0.25, 0.0))()
    converged = bool(model.solver_converged) and np.isfinite(r)
    return model, mesh_obj, dict(alpha=alpha, dofs=model.u_vec.x.petsc_vec.getSize(),
                                 converged=int(converged), steps=model.time_integrator.last_steady_steps,
                                 res=r, wall_s=wall, case_s=time.perf_counter() - tic_case,
                                 init_steps=init["steps"], init_s=init["wall_s"],
                                 c_l=cl, c_d=cd, c_d_pressure=cd - cdf, c_d_friction=cdf, c_m=cm)


def rss_gib(comm):
    """(total, largest rank) resident memory of the ranks in GiB. Collective."""
    with open("/proc/self/status") as f:
        kib = next(int(line.split()[1]) for line in f if line.startswith("VmRSS:"))
    gib = kib / 2**20
    return comm.allreduce(gib, MPI.SUM), comm.allreduce(gib, MPI.MAX)


def main():
    args = parse_args()
    if args.grids is None:
        args.grids = GRID_FILES[args.mesh_family][1]
    comm = MPI.COMM_WORLD
    if args.plot:
        if comm.rank == 0:
            out = args.figure_dir or os.path.dirname(os.path.abspath(args.plot))
            for path in plot_results(args.plot, out, args.format):
                print("Wrote {}".format(path))
        return

    run_dir = None
    if comm.rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)
        stem = "oat15a_{}_{}_M{}_Re{:g}{}_".format(args.mesh_family, args.model + ("-muscl" if euler_muscl(args) else ""),
                                                   args.mach, args.Re,
                                                   "_" + args.label if args.label else "")
        run_dir = tempfile.mkdtemp(prefix=stem + time.strftime("%Y%m%d-%H%M%S_"), dir=args.out_dir)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({**vars(args), "ranks": comm.size}, f, indent=2)
    run_dir = comm.bcast(run_dir, root=0)
    csv_path = os.path.join(run_dir, "results.csv")
    PETSc.Sys.Print("=" * 72)
    PETSc.Sys.Print("OAT15A: {} grids {}, alpha {} deg, M={}, Re={:g}, T={} K, {} ranks".format(
        args.mesh_family, args.grids, args.alphas, args.mach, args.Re, args.T_inf, comm.size))
    PETSc.Sys.Print("Output: {}".format(run_dir))
    PETSc.Sys.Print("=" * 72)

    rows = []
    for g in args.grids:
        mesh_path = os.path.join(args.mesh_dir, GRID_FILES[args.mesh_family][0].format(g))
        msh, info = load_dolfinx_mesh(mesh_path, comm)
        PETSc.Sys.Print("grid {}: {} cells, {} wall edges, chord {} (file units)".format(
            g, info["n_cells"], info["n_wall_edges"], info["chord"]))
        for alpha in args.alphas:
            PETSc.Sys.Print("-" * 72)
            PETSc.Sys.Print("grid {}  alpha {} deg".format(g, alpha))
            model, mesh_obj, rec = solve_case(msh, args, alpha, info["n_wall_edges"])
            rows.append(dict(grid=g, cells=info["n_cells"], **rec))
            if comm.rank == 0:
                np.savetxt(csv_path, np.array([[r[c] for c in COLUMNS] for r in rows], dtype=float),
                           delimiter=",", comments="", header=",".join(COLUMNS),
                           fmt=[FORMATS.get(c, "%.10f") for c in COLUMNS])
            wall_writer(model, mesh_obj)(os.path.join(run_dir, "wall_grid{}_a{}.csv".format(g, alpha)))
            # the converged state, partition independent: reload it on the same
            # grid with load_checkpoint to post-process (e.g. C_p) later
            save_checkpoint(os.path.join(run_dir, "solution_grid{}_a{}.npz".format(g, alpha)), [model.u_vec],
                            {"mesh_family": args.mesh_family, "grid": g, "mesh": os.path.abspath(mesh_path),
                             "alpha": alpha, "mach": args.mach, "Re": args.Re, "T_inf": args.T_inf,
                             "chord_file_units": info["chord"],
                             "converged": bool(rec["converged"])})
            if args.write_fields:
                field_writer(model, run_dir, prefix="grid{}_a{}_".format(g, alpha))()
            PETSc.Sys.Print("RESULT grid {} alpha {}: converged={} steps={} |R|={:.3e} c_l={:.6f} "
                            "c_d={:.6f} (pressure {:.6f}, friction {:.6f}) c_m={:.6f}  [{:.1f} s, RSS {:.2f} GiB, max rank {:.2f}]"
                            .format(g, alpha, bool(rec["converged"]), rec["steps"], rec["res"], rec["c_l"],
                                    rec["c_d"], rec["c_d_pressure"], rec["c_d_friction"], rec["c_m"], rec["wall_s"],
                                    *rss_gib(comm)))
            # Free the case's memory before the next: the SNES holds the
            # solver's bound-method callbacks at C level, out of the garbage
            # collector's sight, so it must be destroyed for the model to be
            # collected at all; and petsc4py only queues the collected parallel
            # Mats/Vecs as garbage on the mesh's communicator, freed by the
            # collective garbage_cleanup. Without both, every case's model
            # stays in memory.
            model.solver.destroy()
            del model, mesh_obj
            gc.collect()
            PETSc.garbage_cleanup(msh.comm)

    if comm.rank == 0:
        plot_results(csv_path, run_dir, args.format)
    PETSc.Sys.Print("Done. Results in {}".format(run_dir))


if __name__ == "__main__":
    main()
