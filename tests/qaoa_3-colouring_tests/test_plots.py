import pytest
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os.path as path
import numpy as np
from tempfile import NamedTemporaryFile
from qtcorgi import GraphGenerator
from qtcorgi import plots


current_dir = path.dirname(path.realpath(__file__))
data_loc = path.join(current_dir, "data", "figure_data")


def figures_are_simular(reference_image, created_image):

    # TODO find better way to check if image is correct
    plt.clf()
    f, axarr = plt.subplots(1, 2)

    axarr[0].imshow(reference_image)
    axarr[0].set_title("Reference image")

    axarr[1].imshow(created_image)
    axarr[1].set_title("Created image")

    f.suptitle("Check if these plots are simmilar then close")
    f.show()
    while True:
        query = input("\nWere the plots similar?: ")
        answer = query[0].lower()
        if query == "" or not answer in ["y", "n"]:
            print("Please answer with yes or no!")
        else:
            break
    if answer == "y":
        return True
    if answer == "n":
        return False


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

    graph_dict = np.load(path.join(data_loc, "graph_dict.npy"))
    connectivity_labels = ["Low connectivity", "High connectivity", "Highest connectivity"]
    connectivity_colours = ["#1b9e77", "#d95f02", "#7570b3"]

    capmanager.suspend_global_capture(in_=True)
    number_of_gates = plots.NumberOfGates(
        graphs_count_gates_dictionary=graph_dict,
        num_layers=2,
        connectivity_labels=connectivity_labels,
        connectivity_colours=connectivity_colours,
        bar_width=0.3,
    )
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
    connectivity_labels = ["d=2", "d=3", "d=4"]
    connectivity_colours = ["#1b9e77", "#d95f02", "#7570b3"]
    bar_width = 0.115
    yield plots.ProbabilityCorrectSampled(connectivity_labels, connectivity_colours, bar_width)


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
