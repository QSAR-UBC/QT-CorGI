from pennylane import numpy as np


def merge_intermittent_graphs(get_save_location_intermittent, end_index):
    """
    Merge graph info from intermittent saved solutions of graphs

    Args:
        get_save_location_intermittent (func): a function that returns the string of a file to load
        end_index (int): the number of graphs that have been solved for
    Returns:
        qubit_info (dict): Dictionary of qubit solution information of graphs solved previously
        qutrit_info (dict): Dictionary of qutrit solution information of graphs solved previously
    """
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

    for i in range(end_index):
        file_name = get_save_location_intermittent(i)
        graph_info = np.load(file_name, allow_pickle=True).item()

        qubits_info_for_graph = graph_info["qubits"]
        qutrits_info_for_graph = graph_info["qutrits"]

        qubit_info["objective_val"].append(qubits_info_for_graph["objective_val"])
        qubit_info["quad_string_ints"].append(qubits_info_for_graph["quad_string_ints"])
        qubit_info["alpha_mf"].append(qubits_info_for_graph["alpha_mf"])
        qubit_info["params"].append(qubits_info_for_graph["params"])

        qutrit_info["objective_val"].append(qutrits_info_for_graph["objective_val"])
        qutrit_info["trit_string_ints"].append(qutrits_info_for_graph["trit_string_ints"])
        qutrit_info["alpha_mf"].append(qutrits_info_for_graph["alpha_mf"])
        qutrit_info["params"].append(qutrits_info_for_graph["params"])

    return qubit_info, qutrit_info
