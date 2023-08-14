import networkx as nx
import numpy as np
from tqdm import tqdm
import os

dir_name = os.path.dirname(__file__)

unchecked_files_location = os.path.join(dir_name, "isomorphicUnchecked")
checked_files_location = os.path.join(dir_name, "checked3Colourable")


def parse_and_check_nonisomorphic_graphs():
    """
    Checks graphs from http://users.cecs.anu.edu.au/~bdm/data/graphs.html for being 3-colourable
    and saves them to ./checked3Colourable
    """
    __make_folders(checked_files_location)

    files_to_check_1 = [f"graph{i}c.g6" for i in range(2, 10)]
    check_files(files_to_check_1)

    files_to_check_2 = ["graph10c_first.g6"]
    check_files(files_to_check_2)

    files_to_check_3 = ["graph10c_second.g6"]
    check_files(files_to_check_3)


def check_files(files_to_check):
    """
    Graphs are dowloaded from http://users.cecs.anu.edu.au/~bdm/data/graphs.html

    Keyword arguments:
    files_to_check -- list of file names of graph 6 files containing all non-isomorphic
                      connected graphs containing n nodes.

    Writes:
    folder checked3Colourable
        folders number_of_nodes_n
            files connectivity_d.g6
    For the folders number_of_nodes_n 'n' refer to how many nodes graphs contained inside have
    For the files connectivity_d.g6 'd' refers to the average number of edges each node is connected
        (rounded down)
    """

    for file_to_check in tqdm(files_to_check, desc="Working through files", position=0):
        path = os.path.join(unchecked_files_location, file_to_check)

        graphs = nx.read_graph6(path)

        if not type(graphs) is list:
            graphs = [graphs]
        n = graphs[0].number_of_nodes()

        nodes_folder = os.path.join(checked_files_location, f"number_of_nodes_{n}")
        __make_folders(nodes_folder)

        graphs_to_save = {}
        ds = []
        max_edges = get_max_edges(n)

        not_three_col = 0
        for graph in tqdm(graphs, desc=f"Solving graphs from {file_to_check}", 
                          position=1, leave=False):
            three_colourable = is_graph_three_colourable(graph, max_edges)
            if three_colourable:
                d = round((2 * graph.number_of_edges()) / n)

                if d not in ds:
                    ds.append(d)
                    graphs_to_save[d] = []

                graphs_to_save[d].append(graph)
            else:
                not_three_col += 1

        for d in graphs_to_save.keys():
            save_loc = os.path.join(nodes_folder, f"connectivity_{d}.g6")
            file = __open_file(save_loc)

            for graph in graphs_to_save[d]:
                file.write(nx.to_graph6_bytes(graph, header=False))

        print(
            "\nTime taken to check and save {} graphs: {:.2f}s".format(i, time.time() - start_time)
        )
        print(f"Number of non three colourable graphs removed {not_three_col}")
    print("done set")


def get_max_edges(n):
    """
    Gets maximum number of edges for a graph with n nodes to still be three colorable

    Args:
        n (int): number of nodes

    Returns:
        max_edges (int): maximum number of edges for a n node 3-colourable graph
    """
    max_edges = 0
    cols = [0, 0, 0]
    for i in range(n):
        col = cols[i % 3]
        cols[i % 3] += 1
        add = i - col
        max_edges += add
    return max_edges


def is_graph_three_colourable(graph, max_edges):
    """
    Checks if a graph is three colourable

    Args:
        graph (networkx.Graph): graph instance to check if three colourable
        max_edges (int): maximum number of edges for a n node 3-colourable graph

    Returns:
        bool: truth value for "graph is 3-colourable"
    """
    if graph.number_of_edges() > max_edges:
        return False

    return __check_three_colourable(graph, graph.number_of_nodes(), [])


"""Private"""


def __check_three_colourable(graph, num_nodes, colouring):
    current_node = len(colouring)
    if num_nodes == current_node:
        return True
    for colour in [0, 1, 2]:
        # check_valid_addition
        valid = True
        for neighbor in graph.neighbors(current_node):
            if neighbor < current_node and colouring[neighbor] == colour:
                valid = False
                break
        if valid and __check_three_colourable(graph, num_nodes, colouring + [colour]):
            return True
    return False


def __make_folders(save_folder):
    try:
        os.makedirs(save_folder)
    except OSError as error:
        print(f"{save_folder}\nFile already made")
        if __check_continue():
            print("Continuing...")
        else:
            raise error


def __open_file(save_loc):
    try:
        return open(save_loc, "xb")
    except OSError:
        print(f"{save_loc} \nfile already made")
        return open(save_loc, "ab")


def __check_continue():
    finished = False
    to_continue = False

    while not finished:
        ans = input("would you like to continue? (y/n)")
        if ans.lower() in ["y", "yes"]:
            to_continue = True
            finished = True

        elif ans.lower() in ["n", "no"]:
            finished = True

    return to_continue
