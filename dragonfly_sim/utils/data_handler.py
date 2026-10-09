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
        # (eval_idx, D, L, M, nan) tuple layout and eval_idx indexing as
        # FOM_force_coefficients' (eval_idx, c_d, c_l, c_m, nan).
        self.FOM_forces = []

        # Parameter vector histories
        self.parameter_vectors_per_iteration = []

        # Wall times of the full-order model
        self.FOM_walltime = []

        # Number of DoFs in simulation
        self.fe_dofs = 0
        # Number of design parameters in optimization
        self.num_var = 0

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
