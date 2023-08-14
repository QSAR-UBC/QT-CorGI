import os

from qtcorgi import GraphGenerator
from qtcorgi import plots
import matplotlib.pyplot as plt
from jax import numpy as np


class FigureBuilder:
    """
    The class used to replicate figures from IEEE paper:
    Exploring the Potential of Qutrits for Quantum Optimization of Graph Coloring

    Args:
        num_layers (int): number of QAOA layers used to differing graph sizes
        n_for_layers (int): number of nodes for graphs used to test variable number of layers
        test_layers (list): list of number of layers being tested
        graphs_to_generate_dictionary (dict): dictionary describing which graphs to generate
        and solve
        save_location (str): location to save the data
        fig_save_folder (str): folder where figure is saved
        graphs_count_gates_dictionary (dict): dictionary describing which graphs to generate and
            find circuit cost metrics
        opt_connectivity_labels (list): labels for the connectivity of tested graphs
        cost_connectivity_labels (list): labels for the connectivity of graphs used for cost metrics
        connectivity_colours (list): list of colours used in figures
        opt_bar_width (float): width of bars on plotted graph
        cost_bar_width (float): width of bars on plotted graph
    """

    def __init__(
        self,
        num_layers,
        n_for_layers,
        test_layers,
        graphs_to_generate_dictionary,
        save_location,
        fig_save_folder,
        graphs_count_gates_dictionary,
        opt_connectivity_labels,
        cost_connectivity_labels,
        connectivity_colours,
        opt_bar_width,
        cost_bar_width,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.n_for_layers = n_for_layers
        self.test_layers = test_layers
        self.graphs_to_gen_dict = graphs_to_generate_dictionary
        self.save_loc = save_location
        try:
            os.makedirs(fig_save_folder)
        except:
            print("folder already made")
        self.fig_save_folder = fig_save_folder

        graph_generator = GraphGenerator(False)
        graph_generator.add_test_graphs_with_dict(graphs_count_gates_dictionary)
        graphs_dict = graph_generator.get_graphs_all()

        self.solution_prob_plotter = plots.ProbabilityCorrectSampled(
            opt_connectivity_labels, connectivity_colours, opt_bar_width
        )
        self.num_gates_plotter = plots.NumberOfGates(
            graphs_dict, num_layers, cost_connectivity_labels, connectivity_colours, cost_bar_width
        )

    def make_figure_variable_size(self):
        """
        Make a plot of probability of sampling a correct solution using qubit or qutrit algorithms
        on varying graph sizes from data saved in self.save_loc
        """
        plt.clf()

        save_location = os.path.join(self.fig_save_folder, "probSolutionNoFour.pdf")
        data_loc = os.path.join(self.save_loc, "variable_size")
        data = {}
        for n in self.graphs_to_gen_dict.keys():
            data[n] = {}
            for d in self.graphs_to_gen_dict[n].keys():
                data[n][d] = np.load(
                    os.path.join(data_loc, f"graphs_info_n{n}_d{d}.npy"), allow_pickle=True
                ).item()

        self.solution_prob_plotter.save_fig_size_no_fourth(data, save_location)

    def make_figure_variable_size_with_fourth(self):
        """
        Make a plot of probability of sampling a correct solution using qubit or qutrit algorithms
        on varying graph sizes from data saved in self.save_loc

        No post selection of invalid results
        """
        plt.clf()

        save_location = os.path.join(self.fig_save_folder, "probSolutionWithFour.pdf")
        data_loc = os.path.join(self.save_loc, "variable_size")
        data = {}
        for n in self.graphs_to_gen_dict.keys():
            data[n] = {}
            for d in self.graphs_to_gen_dict[n].keys():
                data[n][d] = np.load(
                    os.path.join(data_loc, f"graphs_info_n{n}_d{d}.npy"), allow_pickle=True
                ).item()

        self.solution_prob_plotter.save_fig_size(data, save_location)

    def make_figure_variable_layers(self):
        """
        Make a plot of probability of sampling a correct solution using qubit or qutrit algorithms
        on varying number of QAOA layers from data saved in self.save_loc
        """
        plt.clf()

        save_location = os.path.join(self.fig_save_folder, "probSolutionLayersNoFour.pdf")
        data_loc = os.path.join(self.save_loc, "variable_layers")

        data = {}
        for p in self.test_layers:
            data[p] = {}
            for d in self.graphs_to_gen_dict[self.n_for_layers].keys():
                data[p][d] = np.load(
                    os.path.join(data_loc, f"graphs_info_p{p}_d{d}.npy"), allow_pickle=True
                ).item()

        self.solution_prob_plotter.save_fig_layers_no_fourth(data, save_location)

    def make_figure_variable_layers_with_fourth(self):
        """
        Make a plot of probability of sampling a correct solution using qubit or qutrit algorithms
        on varying number of QAOA layers from data saved in self.save_loc

        No post selection of invalid results
        """
        plt.clf()

        save_location = os.path.join(self.fig_save_folder, "probSolutionLayers.pdf")
        data_loc = os.path.join(self.save_loc, "variable_layers")

        data = {}
        for p in self.test_layers:
            data[p] = {}
            for d in self.graphs_to_gen_dict[self.n_for_layers].keys():
                data[p][d] = np.load(
                    os.path.join(data_loc, f"graphs_info_p{p}_d{d}.npy"), allow_pickle=True
                ).item()

        self.solution_prob_plotter.save_fig_layers(data, save_location)

    def make_figure_number_entangling_graphs(self):
        """
        Make a plot of the number of entangling gates necessary when using qubit or qutrit algorithm
        for varying graph sizes
        """
        plt.clf()

        save_location = os.path.join(self.fig_save_folder, "numberOfEntanglingGates.pdf")
        self.num_gates_plotter.save_fig_entangled(save_location)

    def make_figure_number_entangling_graphs(self):
        """
        Make a plot of the total number of gates necessary when using qubit or qutrit algorithm
        for varying graph sizes
        """
        plt.clf()

        save_location = os.path.join(self.fig_save_folder, "numberOfTotalGates.pdf")
        self.num_gates_plotter.save_fig_total_gates(save_location)
