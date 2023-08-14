import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from .utils import get_inds
from ..qubitQaoa import QubitQaoa
from ..qutritQaoa import QutritQaoa


class ProbabilityCorrectSampled:
    """
    Plots the probability of sampling a correct solution

    Args:
        graphs_count_gates_dictionary (dict): dictionary describing which graphs to generate and
            find circuit cost metrics
        num_layers (int): number of QAOA layers used to differing graph sizes
        cost_connectivity_labels (list): labels for the connectivity of graphs used for cost metrics
        connectivity_colours (list): list of colours used in figures
        bar_width (float): width of bars on plotted graph
    """

    def __init__(self, connectivity_labels, connectivity_colours, bar_width):
        self.bar_width = bar_width
        self.connectivity_labels = connectivity_labels
        self.connectivity_colours = connectivity_colours

    def save_fig_size(self, data, save_loc, save_type=None):
        """
        Plots the probability of sampling a correct solution for graphs of varying size

        Args:
            data (dict): solution data including final samples of qubit and qutrit values
            save_loc (str): PATH to where figure will be saved
        """

        def set_data_func(input_data):
            return self.__set_data(input_data, False, "size")

        self.__save_fig_from_data(
            data, save_loc, set_data_func, "Number of nodes, data includes fourth colour", save_type
        )
        print("\nSaved size figure")

    def save_fig_size_no_fourth(self, data, save_loc, save_type=None):
        """
        Plots the probability of sampling a correct solution for graphs of varying size
        Post selects qubit solutions for only solutions without the fourth state

        Args:
            data (dict): solution data including final samples of qubit and qutrit values
            save_loc (str): PATH to where figure will be saved
        """

        def set_data_func(input_data):
            return self.__set_data(input_data, True, "size")

        self.__save_fig_from_data(data, save_loc, set_data_func, "Number of nodes", save_type)
        print("\nSaved size figure")

    def save_fig_layers(self, data, save_loc, save_type=None):
        """
        Plots the probability of sampling a correct solution with varying number of QAOA layers

        Args:
            data (dict): solution data including final samples of qubit and qutrit values
            save_loc (str): PATH to where figure will be saved
        """

        def set_data_func(input_data):
            return self.__set_data(input_data, False, "layers")

        self.__save_fig_from_data(
            data,
            save_loc,
            set_data_func,
            "Number of QAOA layers, data includes fourth colour",
            save_type,
        )
        print("\nSaved layers figure")

    def save_fig_layers_no_fourth(self, data, save_loc, save_type=None):
        """
        Plots the probability of sampling a correct solution with varying number of QAOA layers
        Post selects qubit solutions for only solutions without the fourth state

        Args:
            data (dict): solution data including final samples of qubit and qutrit values
            save_loc (str): PATH to where figure will be saved
        """

        def set_data_func(input_data):
            return self.__set_data(input_data, True, "layers")

        self.__save_fig_from_data(data, save_loc, set_data_func, "Number of QAOA layers", save_type)
        print("\nSaved layers figure")

    """Private"""

    def __save_fig_from_data(self, data, save_loc, set_data_func, x_label, save_type):
        plt.rcParams.update({"font.size": 15})

        qubits, qubits_er, qutrits, qutrits_er, ns = set_data_func(data)
        legend_bars = self.__set_bar_graph_data(qubits, qutrits, qubits_er, qutrits_er, ns, data)
        first_x = min(data.keys())

        plt.xlabel(x_label)
        plt.ylabel("Avg. prob. sampling a solution")
        # plt.title("Probability of Sampling a Correct Solution", fontweight="bold")

        bar1 = plt.bar(
            [first_x],
            [0],
            self.bar_width,
            fill=False,
            hatch="///",
            label="Qubit",
            edgecolor="#838383",
        )
        bar2 = plt.bar([first_x], [0], self.bar_width, label="Qutrit", color="#838383")

        first_legend = plt.legend(
            handles=[bar1, bar2], bbox_to_anchor=(1.2, 0.75), loc="center", title="Solution method"
        )
        ax = plt.gca().add_artist(first_legend)
        plt.legend(
            handles=legend_bars, bbox_to_anchor=(1.2, 0.25), loc="center", title="Connectivity"
        )
        if save_type == None:
            save_loc_str = str(save_loc)
            if "." not in save_loc_str:
                raise ValueError("save location must have format or include_save_type")
            file_ext_ind = save_loc.rfind(".")
            save_type = save_loc[file_ext_ind + 1 :]
        plt.savefig(save_loc, bbox_inches="tight", format=save_type, dpi=1200)

    def __find_error(self, graph, trit_string):
        cost = 0
        for edge in graph.edges:
            cost += trit_string[edge[0]] == trit_string[edge[1]]
        return cost / len(graph.edges)

    def __get_prob_sol(self, testRun):
        quad_string_ints = testRun["qubits"]["quad_string_ints"]
        trit_string_ints = testRun["qutrits"]["trit_string_ints"]
        graphs = testRun["graphs"]
        n_graphs = len(graphs)

        bit_probs = [0] * n_graphs
        trit_probs = [0] * n_graphs
        for i in tqdm(range(n_graphs), desc=" Getting data", position=2, leave=False):
            graph = graphs[i]
            n_nodes = graph.number_of_nodes()
            n_samples = len(trit_string_ints[i])
            quad_strings = list(
                Counter(
                    QubitQaoa.get_quad_strings_from_ints(quad_string_ints[i], n_nodes)[0]
                ).items()
            )
            trit_strings = list(
                Counter(QutritQaoa.get_trit_strings_from_ints(trit_string_ints[i], n_nodes)).items()
            )
            bit_probs[i] = 0
            trit_probs[i] = 0
            for quad_string in quad_strings:
                if self.__find_error(graph, quad_string[0]) == 0:
                    bit_probs[i] += quad_string[1]
            for trit_string in trit_strings:
                if self.__find_error(graph, trit_string[0]) == 0:
                    trit_probs[i] += trit_string[1]

            bit_probs[i] /= n_samples
            trit_probs[i] /= n_samples
        return bit_probs, trit_probs

    def __remove_bad_colourings(self, qstrings):
        newqstringList = []
        for qs in tqdm(qstrings, desc="Removing fourth colourings", position=2, leave=False):
            newqstrings = []
            for qString in qs:
                quad_string = np.base_repr(qString, base=4)
                if not "3" in quad_string:
                    newqstrings.append(qString)
            newqstringList.append(newqstrings)
        return newqstringList

    def __set_bar_graph_data(self, qubits, qutrits, qubits_er, qutrits_er, params, data):
        data_keys = list(data.keys())
        plt.xticks(data_keys)
        ds = []

        for param in data.keys():
            ds.append(len(data[param]))

        base_position = (-(np.array(ds) - 1 / 2) * self.bar_width) + data_keys

        legend_bars = []
        for i, d in enumerate(params.keys()):
            inds = get_inds(data_keys, params[d])
            qubit_pos = (base_position + (2 * i) * self.bar_width)[inds]
            qutrit_pos = qubit_pos + self.bar_width
            colour = self.connectivity_colours[i]
            error_kw = {"lw": 1, "capsize": 2, "capthick": 1}

            bar = plt.bar(
                qubit_pos[0],
                [0],
                self.bar_width,
                color=colour,
                edgecolor="black",
                label=self.connectivity_labels[i],
            )
            legend_bars.append(bar)
            plt.bar(
                qubit_pos,
                qubits[d],
                self.bar_width,
                yerr=qubits_er[d],
                fill=False,
                hatch="///",
                error_kw=error_kw,
                edgecolor=colour,
            )
            plt.bar(
                qutrit_pos,
                qutrits[d],
                self.bar_width,
                yerr=qutrits_er[d],
                color=colour,
                error_kw=error_kw,
            )
        return legend_bars

    def __set_data(self, data, remove_bad_colourings, data_type):
        qubits = {}
        qubits_er = {}

        qutrits = {}
        qutrits_er = {}
        params = {}
        for param in tqdm(data.keys(), desc=f"Working on {data_type} graphs", position=0):
            for d in tqdm(
                data[param].keys(),
                desc=" Working through connectivity data",
                position=1,
                leave=False,
            ):
                if d not in params.keys():
                    params[d] = []
                    qubits[d] = []
                    qubits_er[d] = []
                    qutrits[d] = []
                    qutrits_er[d] = []
                params[d].append(param)
                if remove_bad_colourings:
                    data[param][d]["qubits"]["quad_string_ints"] = self.__remove_bad_colourings(
                        data[param][d]["qubits"]["quad_string_ints"]
                    )
                qubit_best_error, qutrits_best_error = self.__get_prob_sol(data[param][d])
                qubits[d].append(np.mean(qubit_best_error))
                qubits_er[d].append(np.std(qubit_best_error))

                qutrits[d].append(np.mean(qutrits_best_error))
                qutrits_er[d].append(np.std(qutrits_best_error))
        return qubits, qubits_er, qutrits, qutrits_er, params
