from pennylane import numpy as np
import pennylane as qml
from matplotlib import pyplot as plt
from ..qutritQaoa import QutritQaoa
from ..qubitQaoa import QubitQaoa
import pennylane.numpy as np
from tqdm import tqdm
from .utils import get_inds


class NumberOfGates:
    """
    Plots number of gates necessary to run qubit and qutrit QAOA simulations

    Warning:
        Initiation is slow as data is being gathered

    Args:
        graphs_count_gates_dictionary (dict): dictionary describing which graphs to generate and
            find circuit cost metrics
        num_layers (int): number of QAOA layers used to differing graph sizes
        cost_connectivity_labels (list): labels for the connectivity of graphs used for cost metrics
        connectivity_colours (list): list of colours used in figures
        bar_width (float): width of bars on plotted graph
    """

    def __init__(
        self,
        graphs_count_gates_dictionary,
        num_layers,
        connectivity_labels,
        connectivity_colours,
        bar_width,
    ):
        self.graphs_dict = graphs_count_gates_dictionary
        self.num_layers = num_layers
        self.connectivity_labels = connectivity_labels
        self.connectivity_colours = connectivity_colours
        self.bar_width = bar_width

        self.bit_data, self.trit_data = self.__prepGateNumbers()

    def save_fig_entangled(self, save_loc, save_type=None):
        """
        Creates and saves a plot counting the number of entangling gates comparing both qubit and
        qutrit algorithms.

        Args:
            save_loc (str): PATH to where figure should be saved
        """

        def count_entangled_gates_from_resources(bit_resources, trit_resources):
            return bit_resources.gate_sizes[2], trit_resources.gate_sizes[2]

        self.__save_fig_gate_count(
            save_loc,
            count_entangled_gates_from_resources,
            "Total number of entangling gates",
            save_type,
        )
        print("\nSaved number of entangling gates figure")

    def save_fig_total_gates(self, save_loc, save_type=None):
        """
        Creates and saves a plot counting the total number of gates comparing both qubit and
        qutrit algorithms.

        Args:
            save_loc (str): PATH to where figure should be saved
        """

        def count_total_gates_from_resources(bit_resources, trit_resources):
            bdat = bit_resources.gate_sizes
            tdat = trit_resources.gate_sizes
            return bdat[1] + bdat[2], tdat[1] + tdat[2]

        self.__save_fig_gate_count(
            save_loc, count_total_gates_from_resources, "Total number of gates", save_type
        )
        print("\nSaved total number of gates figure")

    """Private"""

    def __save_fig_gate_count(self, save_loc, gate_count_function, ylabel, save_type):
        bit_gates, trit_gates = self.__get_data_from_resources(gate_count_function)
        plt.rcParams.update({"font.size": 14})

        legend_bars = self.__set_data(bit_gates, trit_gates)

        first_x = min(list(bit_gates.keys()))
        plt.xlabel("Number of nodes")
        plt.ylabel(ylabel)

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
            handles=[bar1, bar2],
            bbox_to_anchor=(0, 0.64),
            loc="upper left",
            title="Solution method",
        )
        ax = plt.gca().add_artist(first_legend)
        plt.legend(
            handles=legend_bars, bbox_to_anchor=(0, 1.01), loc="upper left", title="Connectivity"
        )

        if save_type == None:
            save_loc_str = str(save_loc)
            if "." not in save_loc_str:
                raise ValueError("save location must have format or include_save_type")
            file_ext_ind = save_loc.rfind(".")
            save_type = save_loc[file_ext_ind + 1 :]
        plt.savefig(save_loc, bbox_inches="tight", format=save_type, dpi=1200)

    def __prepGateNumbers(self):
        bit_resources = {}
        trit_resources = {}
        for n in tqdm(self.graphs_dict.keys(), desc="Working on generating cost", position=0):
            bit_resources[n] = {}
            trit_resources[n] = {}

            ds = list(self.graphs_dict[n].keys())
            iterator = tqdm(
                range(len(ds)), desc=f"Working on graphs n={n}", position=1, leave=False
            )
            for i in iterator:
                d = ds[i]
                bit_resources[n][i] = []
                trit_resources[n][i] = []
                for graph in tqdm(
                    self.graphs_dict[n][d],
                    desc=f"Solving graphs n={n}, d={d}",
                    position=2,
                    leave=False,
                ):
                    qubit_qaoa = QubitQaoa(graph)
                    qutrit_qaoa = QutritQaoa(graph)

                    @qml.qnode(qml.device("default.qubit", wires=0))
                    def unitary_circuit_qubit(num_layers):
                        rand_gammas = np.random.rand(num_layers)
                        rand_betas = np.random.rand(num_layers)
                        qubit_qaoa.unitary_circuit(rand_gammas, rand_betas, num_layers)
                        return qml.sample()

                    @qml.qnode(qml.device("default.qutrit", wires=0))
                    def unitary_circuit_qutrit(num_layers):
                        rand_gammas = np.random.rand(num_layers)
                        rand_betas = np.random.rand(num_layers)
                        qutrit_qaoa.unitary_circuit(rand_gammas, rand_betas, num_layers)
                        return qml.sample()

                    bit_resources[n][i].append(
                        qml.specs(unitary_circuit_qubit)(self.num_layers)["resources"]
                    )
                    trit_resources[n][i].append(
                        qml.specs(unitary_circuit_qutrit)(self.num_layers)["resources"]
                    )
        return bit_resources, trit_resources

    def __get_data_from_resources(self, gate_count_function):
        bit_gates = {}
        trit_gates = {}

        for n in self.bit_data.keys():
            bit_gates[n] = []
            trit_gates[n] = []
            for d in self.bit_data[n].keys():
                bit_array = []
                trit_array = []
                bit_data = self.bit_data[n][d]
                trit_data = self.trit_data[n][d]
                for i in range(len(self.bit_data[n][d])):
                    bit_count, trit_count = gate_count_function(bit_data[i], trit_data[i])
                    bit_array.append(bit_count)
                    trit_array.append(trit_count)
                bit_gates[n].append(bit_array)
                trit_gates[n].append(trit_array)
        return bit_gates, trit_gates

    def __set_data(self, bit_vals, trit_vals):
        postitions = list(self.graphs_dict.keys())
        plt.xticks(postitions)
        ds = []
        params = {}

        for param in self.graphs_dict.keys():
            length = len(self.graphs_dict[param])
            ds.append(length)
            for i in range(length):
                if not i in params.keys():
                    params[i] = []
                params[i].append(param)

        base_position = (-(np.array(ds) - 1 / 2) * self.bar_width) + postitions

        num_ds = max(ds)
        legend_bars = []
        for i in range(num_ds):
            inds = get_inds(postitions, params[i])
            qubit_pos = (base_position + (2 * i) * self.bar_width)[inds]
            qutrit_pos = qubit_pos + self.bar_width

            bit_n_gates = np.array([bit_vals[n][i] for n in params[i]])
            trit_n_gates = np.array([trit_vals[n][i] for n in params[i]])

            qubits = np.mean(bit_n_gates, axis=1)
            qubits_er = np.std(bit_n_gates, axis=1)
            qutrits = np.mean(trit_n_gates, axis=1)
            qutrits_er = np.std(trit_n_gates, axis=1)

            colour = self.connectivity_colours[i]
            error_kw = {"lw": 1, "capsize": 1, "capthick": 1}
            plt.bar(
                qubit_pos,
                qubits,
                self.bar_width,
                yerr=qubits_er,
                fill=False,
                hatch="///",
                error_kw=error_kw,
                edgecolor=colour,
            )
            plt.bar(
                qutrit_pos,
                qutrits,
                self.bar_width,
                yerr=qutrits_er,
                color=colour,
                error_kw=error_kw,
            )
            bar = plt.bar(
                qubit_pos[0],
                [0],
                self.bar_width,
                color=colour,
                edgecolor="black",
                label=self.connectivity_labels[i],
                error_kw=error_kw,
            )
            legend_bars.append(bar)
        return legend_bars
