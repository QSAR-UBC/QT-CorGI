from qtcorgi.qaoa3Colouring import GraphGenerator
from qtcorgi.qaoa3Colouring import QubitQaoa
from qtcorgi.qaoa3Colouring import QubitSeparatedParameterQaoa
from qtcorgi.qaoa3Colouring import QutritQaoa
import jax
import os
from pennylane import numpy as np
import copy
import shutil
from tqdm import tqdm

intermittent_folder = ".intermittent"

import qtcorgi.experiments.utils as utils

layers_variable_name = "p"
size_variable_name = "n"


class DataReplicator:
    """
    The class used to replicate data from IEEE paper:
    Exploring the Potential of Qutrits for Quantum Optimization of Graph Coloring

    Args:
        num_layers (int): number of QAOA layers used to differing graph sizes
        num_qutrit_steps (int): maximal number of optimization steps for qutrits
        num_qubit_steps (int): maximal number of optimization steps for qubits
        num_samples (int): number of samples taken for solution
        learning_rate (float): optimization learning rate
        fourth_colour_cost (float): cost for fourth colour for qubit cost Hamiltonian
        n_for_layers (int): number of nodes for graphs used to test variable number of layers
        test_layers (list): list of number of layers being tested
        qubit_convergence_value (float): convergence value for qubit optimization halting
        min_qubit_steps (int): minimum number of optimization steps done for qubit optimization
        save_location (str): location to save the data
        graphs_to_generate_dictionary (dict): dictionary of graphs to generate and solve
        use_separate_params_qubit (bool): boolean deciding if the colouring cost parameter and
            the fourth colour cost parameter are seperated
    """

    def __init__(
        self,
        num_layers,
        num_qutrit_steps,
        num_qubit_steps,
        num_samples,
        learning_rate,
        fourth_colour_cost,
        n_for_layers,
        test_layers,
        save_location,
        qubit_convergence_value,
        min_qubit_steps,
        graphs_to_generate_dictionary,
        use_separate_params_qubit,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.num_qutrit_steps = num_qutrit_steps
        self.num_qubit_steps = num_qubit_steps
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.fourth_colour_cost = fourth_colour_cost
        self.n_for_layers = n_for_layers
        self.test_layers = test_layers
        self.qubit_convergence_value = qubit_convergence_value
        self.min_qubit_steps = min_qubit_steps
        self.save_loc = save_location

        self._variable_size_save_folder = os.path.join(self.save_loc, "variable_size")
        self._variable_layers_save_folder = os.path.join(self.save_loc, "variable_layers")

        graph_generator = GraphGenerator(False)
        graph_generator.add_test_graphs_with_dict(graphs_to_generate_dictionary)
        self.graphs_dict = graph_generator.get_graphs_all()

        self.graphs_to_solve = graphs_to_generate_dictionary

        if use_separate_params_qubit:
            self.qubit_solver = QubitSeparatedParameterQaoa
        else:
            self.qubit_solver = QubitQaoa

    def get_data_variable_size(self):
        """
        Gather and save graph solution data with varying graph size
        """
        save_folder = self._variable_size_save_folder
        self.__check_save_folder(save_folder)
        self.__check_save_folder_and_jax(os.path.join(save_folder, intermittent_folder))

        queue = self.__get_size_queue()

        self.__get_data(save_folder, queue, True)

    def continue_get_data_variable_size(self):
        """
        Continues gathering graph solution data with varying graph size from set start point
        """
        save_folder = self._variable_size_save_folder

        queue = self.__get_size_queue()

        for n in self.finished_size_sets.keys():
            for d in self.finished_size_sets[n]:
                del queue[(n, self.num_layers)][d]
                if len(queue[(n, self.num_layers)]) == 0:
                    del queue[(n, self.num_layers)]

        self.__get_data(save_folder, queue, True)

    def get_data_variable_layers(self, check_repeating_calculation):
        """
        Gather and save graph solution data with varying number of QAOA layers
        """
        save_folder = self._variable_layers_save_folder
        self.__check_save_folder(save_folder)
        self.__check_save_folder_and_jax(os.path.join(save_folder, intermittent_folder))

        queue = self.__get_layers_queue_and_copy(check_repeating_calculation, save_folder, False)

        self.__get_data(save_folder, queue, False)

    def continue_get_data_variable_layers(self, check_repeating_calculation):
        """
        Continues gathering graph solution data with varying number of QAOA layers from
        set start point
        """
        save_folder = self._variable_layers_save_folder

        queue = self.__get_layers_queue_and_copy(check_repeating_calculation, save_folder, True)

        for p in self.finished_layers_sets.keys():
            for d in self.finished_layers_sets[p]:
                del queue[(self.n_for_layers, p)][d]
                if len(queue[(self.n_for_layers, p)]) == 0:
                    del queue[(self.n_for_layers, p)]

        self.__get_data(save_folder, queue, False)

    def set_variable_size_completed(self, finished_vals):
        """
        Records the sets of graphs with varying size whose QAOA solution data has been solved

        Args:
            finished_vals (list): Graph sets that have been solved with data stored
        """
        self.finished_size_vals = finished_vals

    def set_variable_layers_completed(self, finished_vals):
        """
        Records the sets of layers and graphs with whose QAOA solution data has been solved

        Args:
            finished_vals (list): values of layers and connectivity completed so far
        """
        self.finished_layers_vals = finished_vals

    def set_variable_size_intermittent_completed(self, n, d, max_graph_ind):
        """
        Records the set of graphs that was being worked on and needs to be continued from

        Args:
            n (int): number of nodes of graph in graph set
            d (int): connectivity of graphs being solved
            max_graph_ind (list): number of graphs that need to be solved
        """
        jax.config.update("jax_enable_x64", True)

        file_name = f"graphs_info_n{n}_d{d}"
        save_location = os.path.join(self._variable_size_save_folder, f"{file_name}.npy")

        get_intermittent_loc = self.__get_intermittent_loc_function(
            self._variable_size_save_folder, file_name
        )

        if not n in self.finished_size_vals.keys():
            self.finished_size_vals[n] = []
        self.finished_size_vals[n].append(d)

        start_index = int(max_graph_ind) + 1

        qubit_info, qutrit_info = utils.merge_intermittent_graphs(get_intermittent_loc, start_index)

        graphs = self.graphs_dict[n][d]
        for i, graph in tqdm(
            enumerate(graphs[start_index:]),
            desc=f"Finishing n={n}, d={d} graphs from {start_index}",
            total=len(graphs) - start_index,
        ):
            index = start_index + i
            self.__get_graph_info(
                n,
                d,
                self.num_layers,
                get_intermittent_loc,
                index,
                graph,
                qubit_info,
                qutrit_info,
            )
        np.save(
            save_location,
            {"graph": graphs, "qubits": qubit_info, "qutrits": qutrit_info},
        )
        self.__remove_intermittent_files(get_intermittent_loc, len(graphs))

    def set_variable_layers_intermittent_completed(self, p, d, max_graph_ind):
        """
        Records the set of graphs and number of layers used to solve that was being worked on and needs to be continued from

        Args:
            p (int): number of layers used to solve graphs
            d (int): connectivity of graphs being solved
            max_graph_ind (list): number of graphs that need to be solved
        """
        jax.config.update("jax_enable_x64", True)

        file_name = f"graphs_info_p{p}_d{d}"
        save_location = os.path.join(self._variable_layers_save_folder, f"{file_name}.npy")

        get_intermittent_loc = self.__get_intermittent_loc_function(
            self._variable_layers_save_folder, file_name
        )

        if not p in self.finished_layers_vals.keys():
            self.finished_layers_vals[p] = []
        self.finished_layers_vals[p].append(d)

        start_index = int(max_graph_ind) + 1

        qubit_info, qutrit_info = utils.merge_intermittent_graphs(get_intermittent_loc, start_index)

        graphs = self.graphs_dict[self.n_for_layers][d]
        for i, graph in tqdm(
            enumerate(graphs[start_index:]),
            desc=f"Finishing p={p}, d={d} graphs from {start_index}",
            total=len(graphs) - start_index,
        ):
            index = start_index + i
            self.__get_graph_info(
                self.n_for_layers,
                d,
                p,
                get_intermittent_loc,
                index,
                graph,
                qubit_info,
                qutrit_info,
            )

        np.save(
            save_location,
            {"graph": graphs, "qubits": qubit_info, "qutrits": qutrit_info},
        )
        self.__remove_intermittent_files(get_intermittent_loc, len(graphs))

    """Private"""

    def __check_save_folder(self, save_folder):
        try:
            os.makedirs(save_folder)
        except OSError as error:
            print("got an os error")
            print(error)

    def __check_save_folder_and_jax(self, save_folder):
        self.__check_save_folder(save_folder)
        jax.config.update("jax_enable_x64", True)

    def __find_error(self, graph, trit_string):
        cost = 0
        for edge in graph.edges:
            cost += trit_string[edge[0]] == trit_string[edge[1]]
        return cost / len(graph.edges)

    def __get_graph_info(
        self,
        n,
        d,
        num_layers,
        get_save_location_intermittent,
        index,
        graph,
        qubit_info,
        qutrit_info,
    ):
        bitTest = self.qubit_solver(
            graph,
            fourth_colour_cost=self.fourth_colour_cost,
            samples=self.num_samples,
            learning_rate=self.learning_rate,
            convergence_value=self.qubit_convergence_value,
            steps=self.num_qubit_steps,
            min_qubit_steps=self.min_qubit_steps,
            print_position=3,
        )
        doneBitTest = bitTest.qaoa_3_colouring(n_layers=num_layers)

        alpha = self.__find_error(graph, doneBitTest[3])

        tritTest = QutritQaoa(
            graph,
            samples=self.num_samples,
            learning_rate=self.learning_rate,
            steps=self.num_qutrit_steps,
            print_position=3,
        )
        edgeTritTest = tritTest.qaoa_3_colouring(n_layers=num_layers)

        qubit_info["objective_val"].append(doneBitTest[0])
        qubit_info["quad_string_ints"].append(doneBitTest[1])
        qubit_info["alpha_mf"].append(alpha)
        qubit_info["params"].append(doneBitTest[4])

        qubit_on_graph = {
            "objective_val": doneBitTest[0],
            "quad_string_ints": doneBitTest[1],
            "alpha_mf": alpha,
            "params": doneBitTest[4],
        }

        alpha = self.__find_error(graph, edgeTritTest[3])
        qutrit_info["objective_val"].append(edgeTritTest[0])
        qutrit_info["trit_string_ints"].append(edgeTritTest[1])
        qutrit_info["alpha_mf"].append(alpha)
        qutrit_info["params"].append(edgeTritTest[4])

        qutrit_on_graph = {
            "objective_val": edgeTritTest[0],
            "trit_string_ints": edgeTritTest[1],
            "alpha_mf": alpha,
            "params": edgeTritTest[4],
        }

        np.save(
            get_save_location_intermittent(index),
            {
                "graph": self.graphs_dict[n][d][index],
                "qubits": qubit_on_graph,
                "qutrits": qutrit_on_graph,
            },
        )

    def __get_layers_queue_and_copy(self, check_rep_calc, save_folder, continuing_previous_try):
        test_layers_to_run = self.test_layers
        if check_rep_calc:
            test_layers_to_run = copy.deepcopy(self.test_layers)
            try:
                test_layers_to_run.remove(self.num_layers)
                print("num_layers is set and has been removed from what is computed here")
                self.__copy_layers_data(save_folder, continuing_previous_try)
            except ValueError:
                print("num_layers is not in set, you may want to add it")
        queue = {}
        ds = self.graphs_to_solve[self.n_for_layers]

        n = int(self.n_for_layers)
        for p in test_layers_to_run:
            p = int(p)
            queue[(n, p)] = ds
        return queue

    def __copy_layers_data(self, save_folder, continuing_previous_try):
        for entry in os.scandir(self._variable_size_save_folder):
            name = entry.name
            if entry.is_file() and f"graphs_info_n{self.n_for_layers}" in name:
                loc = name.find("_d")
                entry_d = name[loc + 2 : -4]

                if (
                    continuing_previous_try
                    and entry_d in self.finished_layers_vals[self.num_layers]
                ):
                    continue

                entry_location = os.path.join(self._variable_size_save_folder, entry.name)
                save_location = os.path.join(
                    save_folder, f"graphs_info_p{self.num_layers}_d{entry_d}.npy"
                )
                shutil.copy2(entry_location, save_location)

    def __get_size_queue(self):
        queue = {}
        p = int(self.num_layers)
        for n in self.graphs_to_solve.keys():
            n = int(n)
            queue[(n, p)] = self.graphs_to_solve[n]
        return queue

    def __get_data(self, save_folder, queue, is_variable_size):
        variable_name = size_variable_name if is_variable_size else layers_variable_name
        variable_ind = 0 if is_variable_size else 1

        for size_and_layers in tqdm(
            queue.keys(), desc=f"Working on {variable_name}", position=0, total=len(queue)
        ):
            variable_value = size_and_layers[variable_ind]

            progress_bar = tqdm(
                queue[size_and_layers], leave=False, position=1, total=len(queue[size_and_layers])
            )
            for d in progress_bar:
                progress_bar.set_description(
                    f"Processing graphs for {variable_name}={variable_value}, d={d}"
                )

                file_name = f"graphs_info_{variable_name}{variable_value}_d{d}"
                save_location = os.path.join(save_folder, f"{file_name}.npy")

                get_intermittent_loc = self.__get_intermittent_loc_function(save_folder, file_name)
                n = size_and_layers[0]
                p = size_and_layers[1]

                qubit_info = {
                    "objective_val": [],
                    "quad_string_ints": [],
                    "alpha_mf": [],
                    "params": [],
                }
                qutrit_info = {
                    "objective_val": [],
                    "trit_string_ints": [],
                    "alpha_mf": [],
                    "params": [],
                }

                graphs = self.graphs_dict[n][d]
                for index, graph in tqdm(
                    enumerate(graphs),
                    leave=False,
                    desc=f"Solving graph",
                    position=2,
                    total=len(graphs),
                ):
                    self.__get_graph_info(
                        n,
                        d,
                        p,
                        get_intermittent_loc,
                        index,
                        graph,
                        qubit_info,
                        qutrit_info,
                    )
                np.save(
                    save_location,
                    {"graphs": graphs, "qubits": qubit_info, "qutrits": qutrit_info},
                )
                self.__remove_intermittent_files(get_intermittent_loc, len(graphs))

    def __get_intermittent_loc_function(self, save_folder, file_name):
        intermittent_folder_loc = os.path.join(save_folder, intermittent_folder)
        return lambda ind: os.path.join(intermittent_folder_loc, f"{file_name}_graph{ind}.npy")

    @staticmethod
    def __remove_intermittent_files(get_intermediate_file_loc, number_of_files):
        for index in range(number_of_files):
            os.remove(get_intermediate_file_loc(index))
