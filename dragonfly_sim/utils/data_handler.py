import os
import numpy as np
from matplotlib import pyplot as plt

def split_list_of_tuples_into_separate_lists(inp_list, iter_acceptance_list=None):
    list_of_lists = []
    for i in range(len(inp_list[0])):
        single_list_i = [element[i] for element in inp_list]

        if i == 0 and iter_acceptance_list is not None:
            elements_to_include = np.isin(single_list_i, iter_acceptance_list)

        if iter_acceptance_list is not None:
            single_list_i = list(np.array(single_list_i)[elements_to_include])

        list_of_lists += [single_list_i]
    return list_of_lists


class DataStore():
    def __init__(self):
        # Initialize arrays that store outputs

        # Relative residual norms eta = ||r||/||u||. These, not absolute norms,
        # are what to plot or threshold on: eta is free of the sqrt(n_dofs) and
        # state-magnitude factors that make absolute norms incomparable
        # between meshes.
        self.fom_relative_residuals = []

        # objective function histories
        self.FOM_force_coefficients = []
        # Dimensional (unnormalized) FOM lift/drag/moment forces, same
        # (eval_idx, D, L, M_1, ...) tuple layout and eval_idx indexing as
        # FOM_force_coefficients' (eval_idx, c_d, c_l, c_m_1, ...), with one
        # moment entry per moment of the postprocessor, in its order.
        self.FOM_forces = []

        # Parameter vector histories
        self.parameter_vectors_per_iteration = []

        # Wall times of the full-order model
        self.FOM_walltime = []

        # POD reduced-order model (core/reduced_order_model.py), per ROM solve:
        # wall time, eta, (eval_idx, c_d, c_l, c_m_1, ...) like
        # FOM_force_coefficients, whether eta met the acceptance threshold,
        # whether the LSPG solver itself is considered converged (ROM_lspg_converged),
        # and (eval_idx, dc_d, dc_l, dc_m_1, ...) ROM - FOM when the FOM also ran;
        # the snapshot singular values after every snapshot added
        self.ROM_walltime = []
        self.rom_relative_residuals = []
        self.ROM_force_coefficients = []
        self.ROM_meets_threshold = []
        self.ROM_lspg_converged = []
        self.rom_coefficient_errors = []
        self.global_snapshot_sing_vals_per_iteration = []

        # Per adjoint solve (one per output row of a derivative evaluation):
        # (eval_idx, 'rom' or 'fom', wall time). 'rom' is the reduced adjoint
        # of an accepted ROM solution, 'fom' the full-order adjoint.
        self.adjoint_solves = []

        # Per ROM adjoint, when DG_windtunnel_model's compute_all_adjoint_errors is on:
        # (eval_idx, output row, ||psi_rom - psi_fom||_2, relative to ||psi_fom||_2,
        # ||psi_fom||_2, whether the ROM solve was accepted, wall time of the reduced
        # adjoint solve), psi_fom being the full-order adjoint at the same state and
        # right-hand side. For an accepted solve the reduced adjoint is the one the optimizer
        # used; for a rejected one the full-order adjoint is used, and the reduced adjoint of
        # the rejected solve is evaluated at that full-order state. The row numbers the adjoint
        # solves of one evaluation in CSDL's order (objective and constraint rows).
        self.adjoint_errors = []

        # With every adjoint_errors entry, the error the ROM adjoint puts into the derivative,
        # which the distance between the adjoint vectors does not measure (a reduced adjoint can
        # be far from the full-order one and still give its derivative): (eval_idx, output row,
        # whether the ROM solve was accepted, {input: (|c - c_fom|, relative to |c_fom|, |c_fom|,
        # nan)}), c = (dR/dinput)^T psi the adjoint's contribution to the derivative with respect
        # to the model input ('alpha'; 'mesh_node_motions' in the mesh coordinate space, before
        # the mesh warping and shape parameterization), the 2-norm over the input.
        self.adjoint_derivative_errors = []

        # Phase timing (utils/solver_profiling.py), when profiling is on: per
        # forward evaluation ('forward': all of solve_residual_equations,
        # containing the 'rom' and 'fom' flow solves, which are also recorded
        # on their own) and per adjoint solve ('adjoint_rom' / 'adjoint_fom': the
        # one the optimizer used; with compute_all_adjoint_errors also the reference
        # solve of the other kind, 'adjoint_fom_reference' / 'adjoint_rom_reference'),
        # (eval_idx, kind, {phase: (self time, total time, calls)}), the
        # slowest rank's times in seconds. Phases are named "group:name";
        # self times partition a record, totals include nested phases.
        self.phase_times = []

        # ROM - FOM state errors at the evaluations where the FOM also ran:
        # (eval_idx, value), the L2 norm over the domain and over the wall of
        # u_FOM - u_ROM, absolute and relative to the L2 norm of u_FOM
        self.absolute_l2_errors_per_iteration = []
        self.relative_l2_errors_per_iteration = []
        self.absolute_l2_bdry_errors_per_iteration = []
        self.relative_l2_bdry_errors_per_iteration = []

        # Every output quantity of the postprocessor by name ("L", "D", "c_l",
        # "c_d", "M_<name>", "c_m_<name>"): {name: [(eval_idx, value), ...]}
        # for the full-order and the reduced-order solution of an evaluation.
        # ROM - FOM errors follow from output_errors().
        self.FOM_outputs = {}
        self.ROM_outputs = {}

        # Hyper-reduction comparison (core/hyper_reduction.py, DG_windtunnel_model's
        # hyper_reduction): DEIM and MDEIM solve every design the ROM solves, on its basis
        # and from its warm start, beside it; the optimizer never uses them. Per method
        # (<M> = DEIM / MDEIM, <m> its lower case), per solve:
        #   <M>_walltime            (eval_idx, online solve s, offline s: basis, points,
        #                           sampled assembly set-up and operators)
        #   <m>_relative_residuals  (eval_idx, eta)
        #   <M>_force_coefficients  (eval_idx, c_d, c_l, c_m_1, ...)
        #   <M>_meets_threshold, <M>_lspg_converged
        #                           (eval_idx, bool), the ROM's criteria
        #   <m>_iterations          (eval_idx, LSPG iterations)
        #   <m>_coefficient_errors  (eval_idx, (dc_d, dc_l, dc_m...) against the FOM or None
        #                           where the FOM did not run, (...) against the ROM)
        #   <m>_l2_errors           (eval_idx, |u - u_FOM|, relative, the same on the wall,
        #                           relative, and |u - u_ROM| relative to |u_ROM|), the L2
        #                           norms of compute_L2_norm; nan where the FOM did not run
        #   <m>_operator_info       (eval_idx, {n_modes, n_points, sampled_cells,
        #                           assembled_cells, total_cells, conditioning, n_snapshots})
        #   <m>_validation          (eval_idx, {...}) with validate=True
        #                           (utils/hyper_reduction_validation.py)
        #   <M>_outputs             {name: [(eval_idx, value)]} like ROM_outputs
        # per adjoint solve (output row) of a design it solved:
        #   <m>_adjoint_errors      (eval_idx, row, |psi - psi_FOM|, relative to |psi_FOM|,
        #                           |psi - psi_ROM| / |psi_ROM| or nan, wall time), at the
        #                           optimizer's state
        #   <m>_adjoint_derivative_errors
        #                           (eval_idx, row, {input: (|c - c_FOM|, relative to |c_FOM|,
        #                           |c_FOM|, |c - c_ROM| / |c_ROM| or nan)}), the errors of its
        #                           contribution c to the derivative, as adjoint_derivative_errors
        # and per FOM solve it harvested Jacobians from:
        #   <m>_snapshots           (eval_idx, offered, added, stored)
        # Their phases are in phase_times as '<m>_offline', '<m>' and 'adjoint_<m>'; the
        # derivative contributions of every compared adjoint as 'adjoint_contributions'.
        for m in ("deim", "mdeim"):
            M = m.upper()
            for name in (M + "_walltime", m + "_relative_residuals", M + "_force_coefficients",
                         M + "_meets_threshold", M + "_lspg_converged", m + "_iterations",
                         m + "_coefficient_errors", m + "_l2_errors", m + "_operator_info",
                         m + "_validation", m + "_adjoint_errors", m + "_adjoint_derivative_errors",
                         m + "_snapshots"):
                setattr(self, name, [])
        self.DEIM_outputs = {}
        self.MDEIM_outputs = {}

        # Number of DoFs in simulation
        self.fe_dofs = 0
        # Number of design parameters in optimization
        self.num_var = 0

    def record_outputs(self, kind, eval_idx, values):
        """Append the named output values of one evaluation; kind is 'FOM', 'ROM',
        'DEIM' or 'MDEIM'."""
        store = getattr(self, kind + "_outputs")
        for name, value in values.items():
            store.setdefault(name, []).append((eval_idx, float(value)))

    def record_phase_times(self, kind, eval_idx, phases):
        """Append the phase timing {phase: (self, total, calls)} of one solve."""
        self.phase_times.append((eval_idx, kind, dict(phases)))

    def phase_time_series(self, kind, by_group=False):
        """(eval_idx array, {phase: self-time array}) of the records of `kind`, 0 where a
        phase did not occur in a record; by_group sums the phases of a group (the name before
        the first ':'). Self times, so the phases of a record add up to its time."""
        records = [(i, p) for i, k, p in self.phase_times if k == kind]
        key = (lambda n: n.split(":", 1)[0]) if by_group else (lambda n: n)
        names = sorted({key(n) for _, p in records for n in p})
        series = {n: np.zeros(len(records)) for n in names}
        for r, (_, p) in enumerate(records):
            for n, (self_time, _, _) in p.items():
                series[key(n)][r] += self_time
        return np.array([i for i, _ in records], dtype=int), series

    @property
    def output_names(self):
        return list(self.ROM_outputs) if self.ROM_outputs else list(self.FOM_outputs)

    def output_errors(self, name, relative=True, kind="ROM", reference="FOM"):
        """(eval_idx, error) arrays of kind - reference (divided by |reference| if
        relative; ROM - FOM by default, kind and reference among 'FOM', 'ROM',
        'DEIM', 'MDEIM') of the output `name`, for the evaluations that have both.
        A repeated eval_idx in the reference record (the baseline solve) keeps its
        last entry."""
        fom = {int(i): v for i, v in getattr(self, reference + "_outputs").get(name, [])}
        pairs = [(int(i), v, fom[int(i)]) for i, v in getattr(self, kind + "_outputs").get(name, [])
                 if int(i) in fom]
        if not pairs:
            return np.zeros(0, dtype=int), np.zeros(0)
        idx, rom, fom_vals = (np.array(c) for c in zip(*pairs))
        err = rom - fom_vals
        return idx.astype(int), (err / np.abs(fom_vals) if relative else err)

    def compile_store_in_dict(self):
        data_dict = {}
        for data in vars(self):
            # print(data)
            # print(self.__dict__[str(data)])
            data_dict[str(data)] = self.__dict__[str(data)]
        return data_dict

    def write_store_to_numpy_file(self, save_folder="Result_dicts", save_filename="POD_deployment.npy"):
        save_location = save_folder + "/" + save_filename
        os.makedirs(save_folder, exist_ok=True)
        # create dictionary with all data
        data_dict = self.compile_store_in_dict()
        # save data_dict to location
        np.save(save_location, data_dict)
        print("Saved data dictionary to {}".format(save_location))

    def open_dict_and_import_data(self, file_folder="Result_dicts", filename="POD_deployment.npy"):
        file_location = file_folder + "/" + filename
        data_dict = np.load(file_location, allow_pickle=True)
        data_dict = data_dict.item()
        for key, value in data_dict.items():
            setattr(self, key, value)

        # TODO: Add routine to unpack input list(s) for plotting, move actual plotting routines to separate utility file


def plot_walltime_boxplots(data_stores, data_legend, yrange=(0, 3), show_yaxis_label=True):
    # One boxplot of the full-order-model wall times per data store, with a
    # horizontal line at the mean wall time pooled over all data stores
    fom_walltime_list = []
    for data_store in data_stores:
        _, FOM_walltime = zip(*data_store.FOM_walltime)
        fom_walltime_list += [np.array(FOM_walltime)]

    fig = plt.figure(figsize=(2.2,3.))
    ax = fig.add_subplot(111)

    fom_walltime_mean = np.mean(np.concatenate(fom_walltime_list))
    ax.plot([0.5, len(data_stores)+0.5], [fom_walltime_mean, fom_walltime_mean], 'k', linewidth=3)

    ax.boxplot(fom_walltime_list, widths=0.8)

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    # ax.set_yscale("log")
    ax.set_xticks([y + 1 for y in range(len(fom_walltime_list))],
                labels=data_legend)
    # ax.set_xlabel(r'Solver')
    if show_yaxis_label:
        ax.set_ylabel(r'Total wall time [s]')
    # else:
    #     # frame = plt.gca()
    #     ax.axes.yaxis.set_ticklabels([])

    ax.tick_params(axis='x', labelrotation=45)

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_horizontalalignment('center')

    ax.set_ylim(yrange[0], yrange[1])

    plt.subplots_adjust(left=0.23, right=0.955, bottom=0.19, top=0.95)
    # plt.rcParams.update({'font.size': 22})
    plt.show()


# def plot_force_coefficients(data_stores)
