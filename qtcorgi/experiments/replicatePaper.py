import yaml
from qtcorgi.experiments import paper_IEEE_replicator
import os

from .paper_IEEE_replicator import intermittent_folder

config_name = "configIEEE.yml"

config_save_name = "configurations.yml"


def replicate_IEEE_paper(**kwargs):
    """
    Replicate data and figures of submission:

    Keyword arguments:
        save_location (str): location to save the data folder (default "here", saves to this folder)
        save_folder_name (str): data folder name (default "IEEE_submission")
        num_layers (int): number of QAOA layers used to differing graph sizes (default 3)
        num_qutrit_steps (int): maximal number of optimization steps for qutrits (default 50)
        num_qubit_steps (int): maximal number of optimization steps for qubits (default 1000)
        num_samples (int): number of samples taken for solution (default 1000)
        learning_rate (float): optimization learning rate (default 0.01)
        fourth_colour_cost (float): cost for fourth colour for qubit cost Hamiltonian (default 2)
        qubit_convergence_value (float): convergence value for qubit optimization halting
            (default 0.001)
        graphs_to_generate_dictionary (dict): dictionary describing which graphs to generate and
            solve (default found in configIEEE.yml)
        n_for_layers (int): number of nodes for graphs used to test variable number of layers
            (default 7)
        test_layers (list): list of number of layers being tested (default [1,2,3,4,5])
        opt_connectivity_labels (list): labels for the connectivity of tested graphs
            (default found in configIEEE.yml)
        cost_connectivity_labels (list): labels for the connectivity of graphs used for cost metrics
            (default found in configIEEE.yml)
        graphs_count_gates_dictionary (dict): dictionary describing which graphs to generate and
            find circuit cost metrics (default found in configIEEE.yml)
        min_qubit_steps (int): minimum number of optimization steps done for qubit optimization
            (default 200)
        fig_save_folder (str): folder where figure is saved (default "./figs")
        use_separate_params_qubit (bool): boolean deciding if the colouring cost parameter and the
            fourth colour cost parameter are seperated (default True)
        connectivity_colours (list): list of colours used in figures
            (default found in configIEEE.yml)
        opt_bar_width (float): width of bars on plotted graph (default 0.115)
        cost_bar_width (float): width of bars on plotted graph (default 0.3)
    """
    with open(os.path.join(os.path.dirname(__file__), config_name), "rb") as file:
        conf = yaml.safe_load(file.read())  # load the config file
    conf.update(kwargs)

    conf["save_location"] = __update_save_location(conf)
    __print_configurations(conf)

    ieee_data_replicator = paper_IEEE_replicator.DataReplicator(**conf)

    ieee_data_replicator.get_data_variable_size()
    ieee_data_replicator.get_data_variable_layers()

    ieee_figure_builder = paper_IEEE_replicator.FigureBuilder(**conf)

    ieee_figure_builder.make_figure_number_entangling_graphs()

    ieee_figure_builder.make_figure_variable_size()
    ieee_figure_builder.make_figure_variable_layers()

    print("\nFINISHED!")


def __update_save_location(conf):
    try:
        save_loc = conf["save_location"]
        if save_loc == "here":
            save_loc = os.path.dirname(__file__)
        save_loc = os.path.join(save_loc, "data", f"{conf['save_folder_name']}")

        save_location = save_loc
        os.makedirs(save_location)
    except:
        i = 1
        found_valid_save_loc = False
        while not found_valid_save_loc:
            try:
                save_location = f"{save_loc}_({i})"
                os.makedirs(f"{save_loc}_({i})")
                found_valid_save_loc = True
            except:
                i += 1
    finally:
        with open(os.path.join(save_location, config_save_name), "w") as file:
            yaml.dump(conf, file)
        print(f"Data and figures will be saved to {save_location}")
        return save_location


def continue_replicate_IEEE_paper(save_folder):
    """
    Continue data collection for replicating data and figures of submission:

    Keyword arguments:
        save_folder (string): Name of folder from which data collection should continue from.
                    Folder is saved in qtcog/experiments/data/
    """
    save_location = os.path.join(os.path.dirname(__file__), "data", save_folder)
    continue_replicate_IEEE_paper_from_location(save_location)


def continue_replicate_IEEE_paper_from_location(save_location):
    """
    Continue data collection for replicating data and figures of submission,
    exact path to save_folder given:

    Keyword arguments:
        save_location (string): path to save_folder to continue from.
    """
    try:
        with open(os.path.join(save_location, config_save_name), "rb") as file:
            conf = yaml.safe_load(file.read())  # load the config file
    except FileNotFoundError as error:
        print(f"There is no file {save_location}")
        print(
            """Make sure inputs are correct, do you need to call 
              continue_replicate_IEEE_paper or continue_replicate_IEEE_paper_from_location"""
        )
        raise error

    data_folders = os.listdir(save_location)

    conf["save_location"] = save_location

    ieee_data_replicator = paper_IEEE_replicator.DataReplicator(**conf)

    if not "variable_layers" in data_folders:
        set_completed_func = ieee_data_replicator.set_variable_size_completed
        set_intetermittent_func = ieee_data_replicator.set_variable_size_intermittent_completed

        data_location = os.path.join(save_location, "variable_size")
        __check_for_files(data_location, set_completed_func, set_intetermittent_func)

        ieee_data_replicator.get_data_variable_size()

    else:
        set_completed_func = ieee_data_replicator.set_variable_layers_completed
        set_intetermittent_func = ieee_data_replicator.set_variable_layers_intermittent_completed

        data_location = os.path.join(save_location, "variable_layers")
        __check_for_files(data_location, set_completed_func, set_intetermittent_func)

    ieee_data_replicator.get_data_variable_layers()

    ieee_figure_builder = paper_IEEE_replicator.FigureBuilder(**conf)

    ieee_figure_builder.make_figure_variable_size()
    ieee_figure_builder.make_figure_variable_layers()
    ieee_figure_builder.make_figure_number_entangling_graphs()


"""Private"""


def __check_for_files(data_location, set_completed_func, set_intetermittent_func):
    data_files = os.listdir(data_location)
    finished_vals = {}

    for file in data_files:
        if file[0] == ".":
            continue

        loc = file.rfind("_d")
        variable = int(file[13:loc])
        d = int(file[loc + 2 : -4])

        if not variable in finished_vals.keys():
            finished_vals[variable] = []
        finished_vals[variable].append(d)

    set_completed_func(finished_vals)

    data_files_intermittent = os.listdir(os.path.join(data_location, intermittent_folder))

    intermittent = False
    graph_numbers = []
    for file in data_files_intermittent:
        loc1 = file.rfind("_d")
        loc2 = file.rfind("_graph")
        variable = int(file[13:loc1])
        d = int(file[loc1 + 2 : loc2])

        pre_checked = variable in finished_vals.keys() and d in finished_vals[variable]
        if not pre_checked:
            if not intermittent:
                expected_variable = variable
                expected_d = d
                intermittent = True
            elif variable != expected_variable or d != expected_d:
                raise Exception("Found 2 different uncompleted sets of graphs.")
            graph_number = file[loc2 + 6 : -4]
            graph_numbers.append(graph_number)

    if intermittent:
        max_graph_ind = int(max(graph_numbers))
        if max_graph_ind + 1 != len(graph_numbers):
            raise Exception("Missing or wrong number of graphs for uncompleted sets.")

        set_intetermittent_func(expected_variable, d, max_graph_ind)


def __print_configurations(conf):
    print("current configuration settings are as follows:")
    for i, key in enumerate(conf.keys()):
        if i % 5 == 0:
            print()
        print(f"{i+1}: {key} = {conf[key]}")
    print()
