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
        # and (eval_idx, dc_d, dc_l, dc_m_1, ...) ROM - FOM when the FOM also ran;
        # the snapshot singular values after every snapshot added
        self.ROM_walltime = []
        self.rom_relative_residuals = []
        self.ROM_force_coefficients = []
        self.ROM_meets_threshold = []
        self.rom_coefficient_errors = []
        self.global_snapshot_sing_vals_per_iteration = []

        # Per adjoint solve (one per output row of a derivative evaluation):
        # (eval_idx, 'rom' or 'fom', wall time). 'rom' is the reduced adjoint
        # of an accepted ROM solution, 'fom' the full-order adjoint.
        self.adjoint_solves = []

        # Every output quantity of the postprocessor by name ("L", "D", "c_l",
        # "c_d", "M_<name>", "c_m_<name>"): {name: [(eval_idx, value), ...]}
        # for the full-order and the reduced-order solution of an evaluation.
        # ROM - FOM errors follow from output_errors().
        self.FOM_outputs = {}
        self.ROM_outputs = {}

        # Number of DoFs in simulation
        self.fe_dofs = 0
        # Number of design parameters in optimization
        self.num_var = 0

    def record_outputs(self, kind, eval_idx, values):
        """Append the named output values of one evaluation; kind is 'FOM' or 'ROM'."""
        store = {"FOM": self.FOM_outputs, "ROM": self.ROM_outputs}[kind]
        for name, value in values.items():
            store.setdefault(name, []).append((eval_idx, float(value)))

    @property
    def output_names(self):
        return list(self.ROM_outputs) if self.ROM_outputs else list(self.FOM_outputs)

    def output_errors(self, name, relative=True):
        """(eval_idx, error) arrays of ROM - FOM (divided by |FOM| if relative)
        of the output `name`, for the evaluations that have both. A repeated
        eval_idx in the FOM record (the baseline solve) keeps its last entry."""
        fom = {int(i): v for i, v in self.FOM_outputs.get(name, [])}
        pairs = [(int(i), v, fom[int(i)]) for i, v in self.ROM_outputs.get(name, []) if int(i) in fom]
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
