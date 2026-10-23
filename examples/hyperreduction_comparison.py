"""
Summary of a hyper-reduction comparison: the DataStore saved by an optimization run with
DG_windtunnel_model(hyper_reduction=HyperReductionComparison(...)) (e.g.
wing_opt_rans_pod.py --hyper-reduction both), POD ROM vs DEIM vs MDEIM per evaluation.

    python hyperreduction_comparison.py Result_dicts/SLSQP_POD_<suffix>.npy [--figure out.png]

Prints, per method, the relative residual, LSPG iterations, online and offline wall time,
the errors against the FOM (where it ran) and the ROM, the operator sizes, the adjoint errors
and the mean phase breakdown of its solves; with --figure, small multiples of the residual,
the state error and the online wall time over the evaluations.
"""
import argparse
import os

import numpy as np

from dragonfly_sim.utils.data_handler import DataStore

METHODS = ("ROM", "DEIM", "MDEIM")
# the reference categorical palette, fixed order: the colour follows the method
COLORS = {"ROM": "#2a78d6", "DEIM": "#eb6834", "MDEIM": "#1baf7a"}
MARKERS = {"ROM": "o", "DEIM": "s", "MDEIM": "^"}


def load(path):
    ds = DataStore()
    ds.open_dict_and_import_data(os.path.dirname(path) or ".", os.path.basename(path))
    return ds


def series(ds, method):
    """{metric: (eval_idx array, values)} of one method's solves."""
    if method == "ROM":
        walltime = ds.ROM_walltime
        l2 = ds.relative_l2_errors_per_iteration
        return {"eta": np.array(ds.rom_relative_residuals).reshape(-1, 2).T,
                "online": np.array(walltime).reshape(-1, 2).T,
                "l2_fom": np.array(l2).reshape(-1, 2).T}
    m = method.lower()
    walltime = np.array(getattr(ds, method + "_walltime")).reshape(-1, 3)
    l2 = np.array(getattr(ds, m + "_l2_errors")).reshape(-1, 6)
    return {"eta": np.array(getattr(ds, m + "_relative_residuals")).reshape(-1, 2).T,
            "online": walltime[:, :2].T, "offline": walltime[:, [0, 2]].T,
            "l2_fom": l2[:, [0, 2]].T, "l2_rom": l2[:, [0, 5]].T,
            "iterations": np.array(getattr(ds, m + "_iterations")).reshape(-1, 2).T}


def phase_breakdown(ds, kind):
    """Mean self time per phase group over the records of `kind`, largest first."""
    idx, groups = ds.phase_time_series(kind, by_group=True)
    if not len(idx):
        return []
    return sorted(((g, float(v.mean())) for g, v in groups.items()), key=lambda gv: -gv[1])


def report(ds):
    for method in METHODS[1:]:
        m = method.lower()
        rows = getattr(ds, m + "_relative_residuals")
        print("\n=== {}: {} solves".format(method, len(rows)))
        if not rows:
            continue
        s = series(ds, method)
        info = dict(getattr(ds, m + "_operator_info"))
        coef = {e: (f, r) for e, f, r in getattr(ds, m + "_coefficient_errors")}
        print("{:>5} {:>10} {:>5} {:>9} {:>9} {:>10} {:>10} {:>10} {:>12}".format(
            "eval", "eta", "its", "online s", "offln s", "L2/FOM", "L2/ROM", "dc_d FOM", "cells"))
        for k, e in enumerate(s["eta"][0].astype(int)):
            dcd = coef[e][0][0] if coef[e][0] is not None else np.nan
            i = info[e]
            print("{:>5} {:>10.3e} {:>5} {:>9.3f} {:>9.3f} {:>10.3e} {:>10.3e} {:>10.3e} {:>5}/{:<6}".format(
                e, s["eta"][1][k], int(s["iterations"][1][k]), s["online"][1][k], s["offline"][1][k],
                s["l2_fom"][1][k], s["l2_rom"][1][k], dcd, i["sampled_cells"], i["total_cells"]))
        adj = np.array([a[2:] for a in getattr(ds, m + "_adjoint_errors")]).reshape(-1, 4)
        if adj.size:
            print("adjoints: {} solves; relative error vs FOM median {:.3e}, vs ROM median {:.3e}, "
                  "mean wall time {:.3f} s".format(adj.shape[0], np.median(adj[:, 1]),
                                                    np.nanmedian(adj[:, 2]) if np.any(np.isfinite(adj[:, 2]))
                                                    else np.nan, adj[:, 3].mean()))
    print("\n=== mean phase self time per solve (s)")
    for kind in ("rom", "deim", "mdeim", "deim_offline", "mdeim_offline", "adjoint_rom", "adjoint_deim",
                 "adjoint_mdeim"):
        groups = phase_breakdown(ds, kind)
        if groups:
            print("{:>14}: ".format(kind) + ", ".join("{} {:.3g}".format(g, v) for g, v in groups[:6]))


def figure(ds, path):
    import matplotlib
    import matplotlib.ticker
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    panels = (("eta", "relative residual ||R||/||u||"), ("l2_fom", "L2 state error vs FOM (relative)"),
              ("online", "online wall time (s)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), constrained_layout=True)
    for ax, (key, title) in zip(axes, panels):
        for method in METHODS:
            data = series(ds, method).get(key)
            if data is None or not data.size:
                continue
            idx, val = data
            ok = np.isfinite(val) & (val > 0)
            ax.plot(idx[ok], val[ok], color=COLORS[method], marker=MARKERS[method], markersize=5,
                    linewidth=1.5, label=method)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("evaluation")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False)
    fig.savefig(path, dpi=150)
    print("figure written to {}".format(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datastore", help="the DataStore .npy file of the run")
    parser.add_argument("--figure", default=None, help="write the small-multiples figure here")
    args = parser.parse_args()
    store = load(args.datastore)
    report(store)
    if args.figure:
        figure(store, args.figure)
