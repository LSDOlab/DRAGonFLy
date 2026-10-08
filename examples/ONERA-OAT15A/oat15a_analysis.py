"""
Mesh sensitivity study of ONERA's OAT15A airfoil: steady SA-neg RANS at p = 0
on structured grid levels, at the conditions of DPW8 test case 1a: M 0.73,
Re 3e6 (chord), static temperature 271 K, alpha 1.36, 1.50, 2.50, 3.00,
3.10 deg. Grid families (--mesh-family): the Cadence grids, levels 1-6
(default; meshes/Cadence-ONERA-OAT15A_..._Structured/), or the Rizzi grids 1-7
(meshes/ONERA-ONERA-OAT15A-Rizzi/).

The grids are structured multi-block CGNS files one cell thick in span (chord
230 mm); utils/mesh_io_utils reduces each to a 2D quad mesh with chord 1 in
memory. The cases run one at a time, each from the same free stream. A case
that does not converge is recorded as such and the sweep goes on.

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
}
COLUMNS = ("grid", "cells", "dofs", "alpha", "converged", "steps", "res", "wall_s", "case_s",
           "c_l", "c_d", "c_d_pressure", "c_d_friction", "c_m")
FORMATS = {"grid": "%d", "cells": "%d", "dofs": "%d", "alpha": "%.4f", "converged": "%d", "steps": "%d",
           "res": "%.6e", "wall_s": "%.1f", "case_s": "%.1f"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh-family", choices=sorted(GRID_FILES), default="cadence",
                   help="structured CGNS grid family: Cadence levels 1-6 or Rizzi grids 1-7")
    p.add_argument("--grids", type=int, nargs="+", default=None,
                   help="grid levels (default: all levels of the family)")
    p.add_argument("--alphas", type=float, nargs="+", default=[1.36, 1.50, 2.50, 3.00, 3.10],
                   help="angles of attack [deg]")
    p.add_argument("--mach", type=float, default=0.73)
    p.add_argument("--Re", type=float, default=3e6, help="Reynolds number based on the chord")
    p.add_argument("--T-inf", type=float, default=271.0,
                   help="static free-stream temperature [K] (Sutherland); DPW8 test case 1a: 271 K")
    p.add_argument("--farfield", choices=("split", "riemann"), default="split")
    p.add_argument("--mesh-dir", default=MESH_DIR)
    # steady solver
    p.add_argument("--cfl0", type=float, default=5.0)
    p.add_argument("--steady-tol", type=float, default=1e-9)
    p.add_argument("--max-steps", type=int, default=1000, help="PTC step budget per case")
    # output
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


def solve_case(msh, args, alpha, n_wall_edges):
    """Build the model on `msh`, check that its wall is the grid's
    `n_wall_edges` Wall edges, and solve to steady state from the free
    stream. Returns (model, mesh_obj, record dict)."""
    tic_case = time.perf_counter()
    mesh_obj, model, _ = build_model(msh, model="sa", mach=args.mach, alpha_deg=alpha, Re=args.Re,
                                     T_inf=args.T_inf, farfield=args.farfield)
    n_wall = count_wall_facets(mesh_obj)
    if n_wall != n_wall_edges:
        raise RuntimeError("the solver's wall has {} facets, the CGNS Wall family {} edges"
                           .format(n_wall, n_wall_edges))
    model.report_positivity_limiter = False
    model.use_ptc = True
    model.ptc_settings = {**model.ptc_settings, "cfl0": args.cfl0}
    model.max_newton_iterations = args.max_steps
    model.residual_conv_limit = args.steady_tol
    model.set_up_solver()
    tic = time.perf_counter()
    r = model.solve_system().norm()
    wall = time.perf_counter() - tic
    cl, cd, cdf, cm = ForceCoefficients(model, mesh_obj, alpha, aero_centre=(0.25, 0.0))()
    converged = bool(model.solver_converged) and np.isfinite(r)
    return model, mesh_obj, dict(alpha=alpha, dofs=model.u_vec.x.petsc_vec.getSize(),
                                 converged=int(converged), steps=model.time_integrator.last_steady_steps,
                                 res=r, wall_s=wall, case_s=time.perf_counter() - tic_case,
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
        stem = "oat15a_{}_sa_M{}_Re{:g}{}_".format(args.mesh_family, args.mach, args.Re,
                                                   "_" + args.label if args.label else "")
        run_dir = tempfile.mkdtemp(prefix=stem + time.strftime("%Y%m%d-%H%M%S_"), dir=args.out_dir)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({**vars(args), "ranks": comm.size}, f, indent=2)
    run_dir = comm.bcast(run_dir, root=0)
    csv_path = os.path.join(run_dir, "results.csv")
    PETSc.Sys.Print("=" * 72)
    PETSc.Sys.Print("OAT15A: {} grids {}, alpha {} deg, M={}, Re={:g}, T={} K, far field {}, {} ranks".format(
        args.mesh_family, args.grids, args.alphas, args.mach, args.Re, args.T_inf, args.farfield, comm.size))
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
                             "farfield": args.farfield, "chord_file_units": info["chord"],
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
