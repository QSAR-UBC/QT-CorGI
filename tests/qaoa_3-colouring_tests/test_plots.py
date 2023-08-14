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


graphs_to_generate = {
    6: {3: 8},
    9: {3: 8, 4: 8},
    12: {3: 8, 4: 8, 6: 8},
    15: {3: 8, 4: 8, 8: 8},
}
graph_generator = GraphGenerator()
graph_generator.add_test_graphs_with_dict(graphs_to_generate)
graph_dict = graph_generator.get_graphs_all()
connectivity_labels = ["Low connectivity", "High connectivity", "Highest connectivity"]
connectivity_colours = ["#1b9e77", "#d95f02", "#7570b3"]
number_of_gates = plots.NumberOfGates(
    graphs_count_gates_dictionary=graph_dict,
    num_layers=2,
    connectivity_labels=connectivity_labels,
    connectivity_colours=connectivity_colours,
    bar_width=0.3,
)


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


class TestNumGatesFigures:
    def test_save_entangled_gate_count_figure(self, suspend_capture):
        plt.clf()
        with NamedTemporaryFile("r+b", delete=True) as fd:
            print("before")
            number_of_gates.save_fig_entangled(fd, "png")
            print("middle")
            created_image = mpimg.imread(fd.name)
            print("after")
        reference_image = mpimg.imread(path.join(data_loc, "num_entangled_gates.png"))

        with suspend_capture:
            figs_simular = figures_are_simular(reference_image, created_image)
        assert figs_simular

    def test_save_total_gate_count_figure(self, suspend_capture):
        plt.clf()
        with NamedTemporaryFile("r+b", delete=True) as fd:
            print("before")
            number_of_gates.save_fig_total_gates(fd, "png")
            print("mid")
            created_image = mpimg.imread(fd)
            print("after")
        reference_image = mpimg.imread(path.join(data_loc, "num_total_gates.png"))

        with suspend_capture:
            figs_simular = figures_are_simular(reference_image, created_image)
        assert figs_simular


@pytest.fixture(scope="module")
def images(pytestconfig):
    connectivity_labels = ["d=2", "d=3", "d=4"]
    connectivity_colours = ["#1b9e77", "#d95f02", "#7570b3"]
    bar_width = 0.115
    prob_cor_sampled = plots.ProbabilityCorrectSampled(
        connectivity_labels, connectivity_colours, bar_width
    )

    capmanager = pytestconfig.pluginmanager.getplugin("capturemanager")
    data_location = path.join(data_loc, "variable_size_data.npy")
    variable_size_data = np.load(data_location, allow_pickle=True).item()

    data_location = path.join(data_loc, "variable_layers_data.npy")
    variable_layers_data = np.load(data_location, allow_pickle=True).item()

    images = {}
    capmanager.suspend_global_capture(in_=True)

    with NamedTemporaryFile("r+b", delete=True) as fd:
        plt.clf()
        prob_cor_sampled.save_fig_size(variable_size_data, fd, "png")
        created_image = mpimg.imread(fd)
    images["prob_correct_var_size_with_fourth"] = created_image

    with NamedTemporaryFile("r+b", delete=True) as fd:
        plt.clf()
        prob_cor_sampled.save_fig_size_no_fourth(variable_size_data, fd, "png")
        created_image = mpimg.imread(fd)
    images["prob_correct_var_size"] = created_image

    with NamedTemporaryFile("r+b", delete=True) as fd:
        plt.clf()
        prob_cor_sampled.save_fig_layers(variable_layers_data, fd, "png")
        created_image = mpimg.imread(fd)
    images["prob_correct_var_layers_with_fourth"] = created_image

    with NamedTemporaryFile("r+b", delete=True) as fd:
        plt.clf()
        prob_cor_sampled.save_fig_layers_no_fourth(variable_layers_data, fd, "png")
        created_image = mpimg.imread(fd)
    images["prob_correct_var_layers"] = created_image

    print("Done setting data")
    capmanager.resume_global_capture()

    yield images


class TestProbabilityCorrectSampledFigures:
    @pytest.mark.parametrize("with_fourth", [True, False])
    def test_save_variable_size_figure(self, with_fourth, suspend_capture, images):
        plt.clf()
        reference_name = "prob_correct_var_size"
        if with_fourth:
            reference_name += "_with_fourth"

        created_image = images[reference_name]
        reference_image = mpimg.imread(path.join(data_loc, f"{reference_name}.png"))

        with suspend_capture:
            figs_simular = figures_are_simular(reference_image, created_image)
        assert figs_simular

    @pytest.mark.parametrize("with_fourth", [True, False])
    def test_save_variable_layers_figure(self, with_fourth, suspend_capture, images):
        plt.clf()
        reference_name = "prob_correct_var_layers"
        if with_fourth:
            reference_name += "_with_fourth"

        created_image = images[reference_name]
        reference_image = mpimg.imread(path.join(data_loc, f"{reference_name}.png"))

        with suspend_capture:
            figs_simular = figures_are_simular(reference_image, created_image)
        assert figs_simular
