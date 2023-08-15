import pytest
import matplotlib.pyplot as plt
import os.path as path
import numpy as np
from tempfile import NamedTemporaryFile
from qtcorgi import plots


current_dir = path.dirname(path.realpath(__file__))
data_loc = path.join(current_dir, "data", "figure_data")


@pytest.fixture
def suspend_capture(pytestconfig):
    class suspend_guard:
        def __init__(self):
            self.capmanager = pytestconfig.pluginmanager.getplugin("capturemanager")

        def __enter__(self):
            self.capmanager.suspend_global_capture(in_=True)

        def __exit__(self, _1, _2, _3):
            self.capmanager.resume_global_capture()

    yield suspend_guard()


@pytest.fixture(scope="module")
def number_of_gates(pytestconfig):
    capmanager = pytestconfig.pluginmanager.getplugin("capturemanager")

    capmanager.suspend_global_capture(in_=True)
    number_of_gates = get_number_of_gates()
    capmanager.resume_global_capture()

    yield number_of_gates


class TestNumGatesFigures:
    def test_save_entangled_gate_count_figure(self, number_of_gates):
        plt.clf()
        with NamedTemporaryFile("r+b", delete=True) as fd:
            number_of_gates.save_fig_entangled(fd, "png")
            created_image = open(fd.name, "rb").read()
        reference_image = open(path.join(data_loc, "num_entangled_gates.png"), "rb").read()

        assert reference_image == created_image

    def test_save_total_gate_count_figure(self, number_of_gates):
        plt.clf()
        with NamedTemporaryFile("r+b", delete=True) as fd:
            number_of_gates.save_fig_total_gates(fd, "png")
            created_image = open(fd.name, "rb").read()
        reference_image = open(path.join(data_loc, "num_total_gates.png"), "rb").read()

        assert reference_image == created_image


@pytest.fixture()
def probability_correct_sampled():
    yield get_probability_correct_sampled()


class TestProbabilityCorrectSampledFigures:
    @pytest.fixture(autouse=True)
    def _get_number_of_gates(self, probability_correct_sampled):
        self._plotter = probability_correct_sampled

    @pytest.fixture()
    @staticmethod
    def variable_size_data():
        data_location = path.join(data_loc, "variable_size_data.npy")
        return np.load(data_location, allow_pickle=True).item()

    @pytest.fixture()
    @staticmethod
    def variable_layers_data():
        data_location = path.join(data_loc, "variable_layers_data.npy")
        return np.load(data_location, allow_pickle=True).item()

    @pytest.mark.parametrize("with_fourth", [True, False])
    def test_save_variable_size_figure(self, with_fourth, variable_size_data, suspend_capture):
        plt.clf()
        reference_name = "prob_correct_var_size"
        with NamedTemporaryFile("r+b", delete=True) as fd:
            with suspend_capture:
                if with_fourth:
                    self._plotter.save_fig_size(variable_size_data, fd, "png")
                    reference_name += "_with_fourth"
                else:
                    self._plotter.save_fig_size_no_fourth(variable_size_data, fd, "png")
            created_image = open(fd.name, "rb").read()
        reference_image = open(path.join(data_loc, f"{reference_name}.png"), "rb").read()

        assert reference_image == created_image

    @pytest.mark.parametrize("with_fourth", [True, False])
    def test_save_variable_layers_figure(self, with_fourth, variable_layers_data, suspend_capture):
        plt.clf()
        reference_name = "prob_correct_var_layers"
        with NamedTemporaryFile("r+b", delete=True) as fd:
            with suspend_capture:
                if with_fourth:
                    self._plotter.save_fig_layers(variable_layers_data, fd, "png")
                    reference_name += "_with_fourth"
                else:
                    self._plotter.save_fig_layers_no_fourth(variable_layers_data, fd, "png")
            created_image = open(fd.name, "rb").read()
        reference_image = open(path.join(data_loc, f"{reference_name}.png"), "rb").read()

        assert reference_image == created_image


def get_number_of_gates():
    graph_dict = np.load(path.join(data_loc, "graph_dict.npy"), allow_pickle=True).item()
    connectivity_labels = ["Low connectivity", "High connectivity", "Highest connectivity"]
    connectivity_colours = ["#1b9e77", "#d95f02", "#7570b3"]

    number_of_gates = plots.NumberOfGates(
        graphs_count_gates_dictionary=graph_dict,
        num_layers=2,
        connectivity_labels=connectivity_labels,
        connectivity_colours=connectivity_colours,
        bar_width=0.3,
    )
    return number_of_gates


def get_probability_correct_sampled():
    connectivity_labels = ["d=2", "d=3", "d=4"]
    connectivity_colours = ["#1b9e77", "#d95f02", "#7570b3"]
    bar_width = 0.115
    return plots.ProbabilityCorrectSampled(connectivity_labels, connectivity_colours, bar_width)


"""Update figures if necessary"""


def update_number_of_gate_figures():
    """Updates number of gate figures to test against, in case of changes"""
    number_of_gates = get_number_of_gates()
    plt.clf()
    number_of_gates.save_fig_entangled(path.join(data_loc, "num_entangled_gates.png"))
    plt.clf()
    number_of_gates.save_fig_total_gates(path.join(data_loc, "num_total_gates.png"))


def update_prob_correct_sampled_graphs():
    """Updates probability correct solution sampled figures to test against, in case of changes"""
    update_variable_size_figures(True)
    update_variable_size_figures(False)
    update_variable_layers_figures(True)
    update_variable_layers_figures(False)


def update_variable_size_figures(with_fourth):
    plt.clf()
    data_location = path.join(data_loc, "variable_layers_data.npy")
    variable_size_data = np.load(data_location, allow_pickle=True).item()

    prob_corr_samp = get_probability_correct_sampled()
    reference_name = "prob_correct_var_size"
    if with_fourth:
        save_fig = prob_corr_samp.save_fig_size
        reference_name += "_with_fourth"
    else:
        save_fig = prob_corr_samp.save_fig_size_no_fourth
    save_fig(variable_size_data, path.join(data_loc, f"{reference_name}.png"))


def update_variable_layers_figures(with_fourth):
    plt.clf()
    data_location = path.join(data_loc, "variable_layers_data.npy")
    variable_size_data = np.load(data_location, allow_pickle=True).item()

    prob_corr_samp = get_probability_correct_sampled()
    reference_name = "prob_correct_var_layers"
    if with_fourth:
        save_fig = prob_corr_samp.save_fig_layers
        reference_name += "_with_fourth"
    else:
        save_fig = prob_corr_samp.save_fig_layers_no_fourth
    save_fig(variable_size_data, path.join(data_loc, f"{reference_name}.png"))
