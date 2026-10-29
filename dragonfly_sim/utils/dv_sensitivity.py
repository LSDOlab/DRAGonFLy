"""Total-derivative norms per individual design parameter.

Motivation
----------
``PODBasis.construct_basis``'s ``local_weighted`` mode weights snapshots by a plain
Euclidean distance across the snapshot parameter vector. When design variables of
different physical character share that vector -- control point motions in metres
alongside angle of attack in radians -- an equal distance in different dimensions
implies a wildly different change in the state, so the weighting ends up dominated
by whichever variable happens to have the loosest bound. The intended fix is a
diagonal weighting matrix on the distance, one weight per design parameter, sized
by how strongly the solution responds to that parameter.

Adjoint-based optimization never forms ``du/dx``, so the stand-in used here is the
total derivative of the OUTPUT quantities, ``dF/dx``, which the gradient evaluation
already computes. This module extracts it and reports, for each individual design
parameter ``j``,

    s_j = || [dF_1/dp_j, ..., dF_m/dp_j] ||_2

over all optimization outputs F (objective and constraints).

Where the numbers come from
---------------------------
``SimulatorBase.compute_optimization_derivatives()`` returns ``df`` of shape
``(1, n_dv)`` and ``dc`` of shape ``(n_c, n_dv)`` -- the total derivatives of the
objective and constraints with respect to the flattened design variable vector, in
PHYSICAL (unscaled) design variable units. modopt applies its own
``o_scaler``/``x_scaler`` only afterwards, so what the simulator hands back is
exactly what is wanted here. ``attach_total_derivative_probe`` wraps that method on
the simulator INSTANCE, so nothing in ``csdl_alpha`` or ``modopt`` is modified and
no additional solves are performed -- it reads the dict that is already being
returned to the optimizer.

``sim.dv_meta`` / ``sim.c_meta`` / ``sim.o_meta`` (built by
``csdl_alpha.backends.simulator.build_opt_metadata``) map each csdl Variable to its
``l_ind:u_ind`` slice of the flat vector, which is what lets every column be
labelled with its design variable and its index inside that variable.

Units: ``param_scales``
-----------------------
If the POD parameter vector is not the design variable vector verbatim,
``param_scales[j]`` is the factor mapping one to the other,
``p_j = param_scales[j] * x_j``, so the reported sensitivities are divided by it:

    dF/dp_j = (dF/dx_j) / param_scales[j]

That puts the printed numbers directly in the units the POD distance is measured
in. In DRAGonFLy the snapshot parameter vector (``DG_windtunnel_model.
_parameter_vector``: the flat shape parameters, then alpha) is the design variable
vector, so ``param_scales`` is normally left out.

Dependencies
------------
Deliberately numpy-only in its numeric core -- no csdl, petsc or dolfinx --
following the same rule as ``ffd_dv_utils``, so it can be exercised without an MPI
environment. ``extract_metadata`` is the single function that touches csdl objects,
and it only reads plain attributes (``.name``, ``.shape``, ``.size``). Callers pass
their own ``printer`` (typically ``PETSc.Sys.Print``) to keep output on rank 0.
"""

import numpy as np
from typing import NamedTuple


class DVSensitivityMeta(NamedTuple):
    """Labels for the rows and columns of the stacked total-derivative matrix.

    param_labels : n_dv strings, one per individual design parameter, in the
        order of the flattened design variable vector
    group_slices : [(name, l_ind, u_ind)] per design VARIABLE, same order
    output_labels: n_out strings, objective first then constraints, matching the
        row order of ``stack_total_derivatives``
    """
    param_labels: list
    group_slices: list
    output_labels: list

    @property
    def n_dv(self):
        return len(self.param_labels)

    @property
    def n_out(self):
        return len(self.output_labels)


def _entry_labels(name, shape, size):
    """One label per entry of a variable, flattened in C order.

    C order is not a free choice: ``build_opt_metadata`` fills the flat vector
    with ``var.value.flatten()``, and ``ffd_dv_utils.build_cp_motion_dv`` builds
    the matching bounds the same way, so entry ``[i, j, k]`` of a design variable
    is at flat index ``ravel_multi_index((i, j, k), shape)``.
    """
    if size == 1:
        return [name]
    shape = tuple(int(s) for s in shape)
    return ["{}[{}]".format(name, ",".join(str(i) for i in np.unravel_index(k, shape)))
            for k in range(size)]


def extract_metadata(sim):
    """Build a DVSensitivityMeta from a csdl_alpha simulator.

    The ONLY function here that touches csdl objects. Everything downstream works
    on numpy arrays and plain python lists.

    Variables created by ``CustomOperation.create_output`` carry no name (it does
    not call ``add_name``), so unnamed outputs fall back to positional labels. Give
    the objective a name with ``add_name`` at the call site if the printout should
    identify it.
    """
    group_slices = []
    param_labels = []
    for k, var in enumerate(sim.dv_meta):
        name = var.name if var.name is not None else "dv{}".format(k)
        l_ind = sim.dv_meta[var]['l_ind']
        u_ind = sim.dv_meta[var]['u_ind']
        group_slices.append((name, l_ind, u_ind))
        param_labels.extend(_entry_labels(name, var.shape, u_ind - l_ind))

    # Sorted by l_ind rather than trusting dict order: the row order of the
    # stacked matrix below is defined by these index ranges (see
    # SimulatorBase._assemble_jacs), not by insertion order.
    output_labels = []
    for var in sorted(sim.o_meta, key=lambda v: sim.o_meta[v]['l_ind']):
        name = var.name if var.name is not None else "objective"
        output_labels.append("{} (objective)".format(name))

    for k, var in enumerate(sorted(sim.c_meta, key=lambda v: sim.c_meta[v]['l_ind'])):
        name = var.name if var.name is not None else "constraint{}".format(k)
        size = sim.c_meta[var]['u_ind'] - sim.c_meta[var]['l_ind']
        output_labels.extend(_entry_labels(name, var.shape, size))

    return DVSensitivityMeta(param_labels=param_labels,
                             group_slices=group_slices,
                             output_labels=output_labels)


def stack_total_derivatives(df, dc):
    """Stack the objective gradient and constraint Jacobian into one (n_out, n_dv).

    ``df`` is ``(1, n_dv)`` and ``dc`` is ``(n_c, n_dv)``, exactly as
    ``compute_optimization_derivatives`` returns them; either may be None (an
    unconstrained problem, or a feasibility problem with no objective). Returns
    None when both are.
    """
    blocks = []
    if df is not None:
        df = np.atleast_2d(np.asarray(df, dtype=np.double))
        blocks.append(df)
    if dc is not None:
        dc = np.atleast_2d(np.asarray(dc, dtype=np.double))
        blocks.append(dc)

    if len(blocks) == 0:
        return None
    if len(blocks) == 2 and blocks[0].shape[1] != blocks[1].shape[1]:
        raise ValueError(
            "objective gradient and constraint Jacobian disagree on the number of "
            "design parameters: {} vs {}".format(blocks[0].shape[1], blocks[1].shape[1]))

    return np.concatenate(blocks, axis=0)


def total_derivative_norms(J, meta=None, param_scales=None, output_values=None):
    """Column and block norms of the stacked total-derivative matrix.

    Parameters
    ----------
    J : (n_out, n_dv) array from stack_total_derivatives
    meta : DVSensitivityMeta, optional -- needed for the labelled block norms
    param_scales : (n_dv,) array, optional. p_j = param_scales[j]*x_j; J is
        divided by it so every reported quantity is per unit of the POD
        parameter, not per unit of the design variable. See the module docstring.
    output_values : (n_out,) array, optional -- the values of the outputs
        themselves, used for the magnitude-normalized variant.

    Returns
    -------
    dict with keys 'J' (the scaled matrix), 'column_norms',
    'column_norms_relative' (None without output_values) and 'block_norms'
    (None without meta).
    """
    J = np.asarray(J, dtype=np.double)
    if J.ndim != 2:
        raise ValueError("J must be 2-D (n_out, n_dv), got shape {}".format(J.shape))
    n_out, n_dv = J.shape

    if param_scales is not None:
        param_scales = np.asarray(param_scales, dtype=np.double)
        if param_scales.shape != (n_dv,):
            raise ValueError("param_scales must have shape ({},), got {}"
                             .format(n_dv, param_scales.shape))
        if np.any(param_scales == 0.):
            raise ValueError("param_scales contains a zero; it is a per-parameter "
                             "unit conversion and must be nonzero")
        J = J / param_scales[None, :]

    column_norms = np.linalg.norm(J, ord=2, axis=0)

    # The outputs are not commensurate -- in the airfoil problem D is O(1e-2) and
    # L is O(4e-1) -- so the raw stacked norm above is dominated by whichever
    # output happens to be largest, and any weight derived from it inherits that
    # bias. Dividing each row by its own output value gives the relative-response
    # variant, which is the scale-free one. Outputs that are exactly zero are left
    # unnormalized rather than blowing the whole vector up to inf.
    column_norms_relative = None
    if output_values is not None:
        output_values = np.asarray(output_values, dtype=np.double).flatten()
        if output_values.shape != (n_out,):
            raise ValueError("output_values must have shape ({},), got {}"
                             .format(n_out, output_values.shape))
        denom = np.where(np.abs(output_values) > 0., np.abs(output_values), 1.)
        column_norms_relative = np.linalg.norm(J / denom[:, None], ord=2, axis=0)

    block_norms = None
    if meta is not None:
        if meta.n_dv != n_dv or meta.n_out != n_out:
            raise ValueError(
                "metadata describes a ({}, {}) problem but J is {}"
                .format(meta.n_out, meta.n_dv, J.shape))
        block_norms = {}
        for i, output_label in enumerate(meta.output_labels):
            for group_name, l_ind, u_ind in meta.group_slices:
                block_norms[(output_label, group_name)] = float(
                    np.linalg.norm(J[i, l_ind:u_ind], ord=2))

    return {'J': J,
            'column_norms': column_norms,
            'column_norms_relative': column_norms_relative,
            'block_norms': block_norms}


def format_total_derivative_norms(norms, meta, header=None, max_entries=64):
    """Render the norms as a printable multi-line table.

    The per-group summary and the per-output x per-group block norms are always
    shown; the full per-parameter listing only when there are at most
    ``max_entries`` design parameters, since a large FFD block would otherwise
    bury the rest of the log.
    """
    column_norms = norms['column_norms']
    relative = norms['column_norms_relative']
    block_norms = norms['block_norms']

    lines = ["=" * 78]
    lines.append("Total-derivative norms per design parameter"
                 + ("" if header is None else " ({})".format(header)))
    lines.append("outputs stacked: " + ", ".join(meta.output_labels))
    lines.append("=" * 78)

    lines.append("per design-variable group, ||dF/dp||_2 over all outputs:")
    lines.append("  {:<24s} {:>5s}  {:>12s} {:>12s} {:>12s}".format(
        "group", "n", "min", "median", "max"))
    for group_name, l_ind, u_ind in meta.group_slices:
        block = column_norms[l_ind:u_ind]
        lines.append("  {:<24s} {:>5d}  {:>12.6e} {:>12.6e} {:>12.6e}".format(
            group_name, u_ind - l_ind, block.min(), np.median(block), block.max()))

    if block_norms is not None:
        lines.append("")
        lines.append("per output x group, ||dF_i/dp_group||_2:")
        group_names = [group_name for group_name, _, _ in meta.group_slices]
        lines.append("  {:<30s}".format("output")
                     + "".join("{:>15s}".format(name) for name in group_names))
        for output_label in meta.output_labels:
            lines.append("  {:<30s}".format(output_label)
                         + "".join("{:>15.6e}".format(block_norms[(output_label, name)])
                                   for name in group_names))

    if meta.n_dv <= max_entries:
        lines.append("")
        header_line = "  {:>5s}  {:<28s} {:>14s}".format("idx", "parameter", "||dF/dp||")
        if relative is not None:
            header_line += " {:>14s}".format("||dF/dp||/|F|")
        lines.append(header_line)
        for j, label in enumerate(meta.param_labels):
            line = "  {:>5d}  {:<28s} {:>14.6e}".format(j, label, column_norms[j])
            if relative is not None:
                line += " {:>14.6e}".format(relative[j])
            lines.append(line)
    else:
        lines.append("")
        lines.append("  ({} design parameters; per-parameter listing suppressed, "
                     "max_entries={})".format(meta.n_dv, max_entries))

    lines.append("=" * 78)
    return "\n".join(lines)


def print_total_derivative_norms(deriv_dict, meta, printer=print, param_scales=None,
                                 header=None, max_entries=64):
    """Compute and print the norms for one ``compute_optimization_derivatives`` result.

    Returns the norms dict, or None when the result carried no derivatives at all.
    """
    J = stack_total_derivatives(deriv_dict.get('df'), deriv_dict.get('dc'))
    if J is None:
        printer("Total-derivative probe: no objective gradient or constraint "
                "Jacobian in this result; nothing to report.")
        return None

    output_values = _stack_output_values(deriv_dict, J.shape[0])
    norms = total_derivative_norms(J, meta=meta, param_scales=param_scales,
                                   output_values=output_values)
    printer(format_total_derivative_norms(norms, meta, header=header,
                                          max_entries=max_entries))
    return norms


def _stack_output_values(deriv_dict, n_out):
    """The objective and constraint VALUES, in the row order of the stacked J.

    Returns None if they are absent or do not add up to n_out, in which case the
    magnitude-normalized column norms are simply skipped -- they are a diagnostic,
    not worth failing the probe over.
    """
    blocks = []
    for key in ('f', 'c'):
        value = deriv_dict.get(key)
        if value is not None:
            blocks.append(np.asarray(value, dtype=np.double).flatten())
    if len(blocks) == 0:
        return None
    stacked = np.concatenate(blocks)
    return stacked if stacked.shape == (n_out,) else None


def attach_total_derivative_probe(sim, printer=print, param_scales=None,
                                  record_to=None,
                                  record_attr='total_derivative_norms_per_iteration',
                                  eval_idx_getter=None, on_norms=None,
                                  max_entries=64):
    """Print the per-parameter total-derivative norms on every gradient evaluation.

    Wraps ``compute_optimization_derivatives`` on the simulator INSTANCE (an
    instance attribute shadowing the class method), which is what keeps this out of
    csdl_alpha and modopt entirely. modopt guards that call with its own warm-``x``
    check, so it fires once per new design point and costs no additional solves --
    the wrapper only reads the dict already being returned.

    Parameters
    ----------
    sim : a csdl_alpha SimulatorBase set up as an optimization problem
    printer : callable taking one string; pass PETSc.Sys.Print under MPI. The
        derivatives are replicated across ranks (the custom operations allreduce
        their jacvec products), so rank-0 printing loses nothing.
    param_scales : (n_dv,) design-variable-to-parameter unit conversion; see the
        module docstring. Its length is checked against the design variable count
        HERE, at attach time, rather than at the first gradient evaluation -- that
        is the guard that catches a design variable being added without the
        snapshot parameter vector being updated together.
    record_to : optional object (a DataStore) with a list attribute named
        ``record_attr``, appended with ``(eval_idx, column_norms)`` per call.
    eval_idx_getter : callable returning the evaluation index these derivatives
        belong to. Without one, the probe counts its own calls.
    on_norms : optional callable receiving the full norms dict after each
        gradient evaluation. This is how a consumer that wants to ACT on the
        sensitivities gets them -- see
        ReducedOrderModel.set_distance_sensitivities, which turns them
        into the diagonal weights for the local_weighted POD distance. Called
        inside the same guard as the reporting, so a consumer that raises
        degrades the diagnostic rather than the optimization.

    Returns
    -------
    callable that detaches the probe.
    """
    if getattr(sim, '_total_derivative_probe_attached', False):
        raise ValueError("a total-derivative probe is already attached to this "
                         "simulator; detach it before attaching another")

    meta = extract_metadata(sim)

    if param_scales is not None:
        param_scales = np.asarray(param_scales, dtype=np.double).flatten()
        if param_scales.shape != (meta.n_dv,):
            raise ValueError(
                "param_scales has {} entries but the problem has {} design "
                "parameters ({})".format(param_scales.shape[0], meta.n_dv,
                                         ", ".join(name for name, _, _ in meta.group_slices)))

    if record_to is not None and not hasattr(record_to, record_attr):
        raise AttributeError("record_to has no attribute '{}' to append to"
                             .format(record_attr))

    original = sim.compute_optimization_derivatives
    call_count = [0]

    def wrapped(*args, **kwargs):
        deriv_dict = original(*args, **kwargs)

        # The probe is a diagnostic bolted onto the optimizer's critical path;
        # it must never be the reason an optimization dies. Report and carry on.
        try:
            eval_idx = eval_idx_getter() if eval_idx_getter is not None else call_count[0]
            norms = print_total_derivative_norms(
                deriv_dict, meta, printer=printer, param_scales=param_scales,
                header="eval_idx {}".format(eval_idx), max_entries=max_entries)
            if norms is not None and record_to is not None:
                getattr(record_to, record_attr).append(
                    (eval_idx, norms['column_norms'].copy()))
            if norms is not None and on_norms is not None:
                on_norms(norms)
        except Exception as exc:
            printer("WARNING: total-derivative probe failed: {!r}".format(exc))

        call_count[0] += 1
        return deriv_dict

    sim.compute_optimization_derivatives = wrapped
    sim._total_derivative_probe_attached = True

    def detach():
        # Deleting the instance attribute re-exposes the class method underneath.
        try:
            del sim.compute_optimization_derivatives
        except AttributeError:
            pass
        sim._total_derivative_probe_attached = False

    return detach


def attach_rom_adjoint_derivative_comparison(sim, flow_model, record_to, printer=print,
                                             record_attr='rom_adjoint_total_derivatives', methods=None):
    """Compare, at every design with a rejected ROM solve, the total derivatives the optimizer
    gets (full-order adjoint) with those the ROM's adjoint would give it.

    Wraps ``compute_optimization_derivatives`` on the simulator INSTANCE, like
    attach_total_derivative_probe (attach this one first: the probe then sees the optimizer's
    derivatives). After the optimizer's evaluation it evaluates the derivatives again inside
    ``flow_model.rom_adjoint_shadow()``, so that the flow model's reverse products use the
    rejected ROM's adjoint (DG_windtunnel_model.rom_adjoint_shadow) and CSDL chains them through
    the mesh warping, the shape parameterization and the explicit output derivatives as it does
    the full-order ones. The second evaluation re-runs the model on the same design (a repeat,
    which takes an evaluation index) and costs one reverse sweep with reduced adjoints. The
    optimizer gets the first evaluation's dict.

    Appends to ``record_to.<record_attr>`` (rank 0 records, every rank computes)
    ``(eval_idx, df_fom, dc_fom, df_rom, dc_rom)``, eval_idx that of the adjoint records of the
    design (adjoint_errors), the derivatives in physical design variable units as
    compute_optimization_derivatives returns them.

    ``methods``: the reduced adjoints to compare, 'rom' and the hyper-reduced comparison's
    'deim' / 'mdeim' (default: 'rom' and the flow model's hyper-reduction methods), one more
    evaluation each; a hyper-reduced method's records go to ``record_to.<method>_adjoint_total_
    derivatives``.
    """
    if methods is None:
        hr = getattr(flow_model, "hyper_reduction", None)
        methods = ('rom',) + (() if hr is None else tuple(hr.models))
    attrs = {m: record_attr if m == 'rom' else m + "_adjoint_total_derivatives" for m in methods}
    for attr in attrs.values():
        if not hasattr(record_to, attr):
            raise AttributeError("record_to has no attribute '{}' to append to".format(attr))
    original = sim.compute_optimization_derivatives

    def copy(v):
        return None if v is None else np.array(v, dtype=float, copy=True)

    def wrapped(*args, **kwargs):
        deriv_dict = original(*args, **kwargs)
        result = {k: copy(v) if k in ('df', 'dc') else v for k, v in deriv_dict.items()}
        # the adjoint records' index of this design, before the shadow evaluations take their own
        eval_idx = flow_model.eval_idx - 1
        for method in methods:
            try:
                if not flow_model.rom_adjoint_shadow_available(method):
                    continue
                with flow_model.rom_adjoint_shadow(method) as failed:
                    shadow = original(*args, **kwargs)
                    missing = failed()
                if missing:
                    printer("{}-adjoint total derivatives: no {} adjoint at eval_idx {}".format(
                        method.upper(), method.upper(), eval_idx))
                    continue
                df_rom, dc_rom = copy(shadow['df']), copy(shadow['dc'])
                if flow_model.mesh.mesh.comm.Get_rank() == 0:
                    getattr(record_to, attrs[method]).append((eval_idx, result['df'], result['dc'], df_rom, dc_rom))
                J_fom, J_rom = stack_total_derivatives(result['df'], result['dc']), \
                    stack_total_derivatives(df_rom, dc_rom)
                rel = np.linalg.norm(J_rom - J_fom, axis=1) / np.maximum(np.linalg.norm(J_fom, axis=1), 1e-300)
                printer("{}-adjoint total derivatives (eval_idx {}): relative error per output {}".format(
                    method.upper(), eval_idx, " ".join("{:.3e}".format(r) for r in rel)))
            except Exception as exc:
                # a diagnostic must not stop the optimization
                printer("WARNING: {}-adjoint total-derivative comparison failed: {!r}".format(method.upper(), exc))
        return result

    sim.compute_optimization_derivatives = wrapped

    def detach():
        try:
            del sim.compute_optimization_derivatives
        except AttributeError:
            pass

    return detach
