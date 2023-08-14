from pathlib import Path
import numpy as np
from numpy import random as rng
import networkx as nx
from .find3ColourableGraphs import checked_files_location
import os


def contains_triangle(G):
    nodes = G.nodes
    for n in nodes:
        for n2 in G.neighbors(n):
            for n3 in G.neighbors(n2):
                if n in G.neighbors(n3):
                    return True
    return False


def find_bipartite(Vr, Vs, d_rs):
    edges = []

    Vs_edge_num = np.zeros(len(Vs))
    acceptable_Vs = len(Vs)

    for v in Vr:
        selected = np.random.choice(acceptable_Vs, d_rs)
        edges += [(v, v2) for v2 in Vs[selected]]

        Vs_edge_num[selected] += 1
        acceptable_Vs = np.where(Vs_edge_num <= d_rs)[0]
        Vs_edge_num[acceptable_Vs]
        if len(acceptable_Vs) == 0 and v != Vr[-1]:
            return False
    return edges


class GraphGenerator:
    """
    Generates sets of 3-colourable graphs

    Args:
        try_load (bool): Indicates weather graphs should be pulled from file
    """

    _file_name = os.path.join(os.path.dirname(__file__), "3_colourable_graphs.npy")

    def __init__(self, try_load=False):
        self._known_graphs = {
            2: {1: 0},
            3: {1: 0, 2: 0},
            4: {2: 0},
            5: {2: 0, 3: 0},
            6: {2: 0, 3: 0, 4: 0},
            7: {2: 0, 3: 0, 4: 0, 5: 0},
            8: {2: 0, 3: 0, 4: 0, 5: 0},
            9: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
            10: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
        }
        if try_load and Path(self._file_name).is_file():
            self._graphs = np.load(self._file_name, allow_pickle=True)
        else:
            self._graphs = {}

    def add_test_graphs_with_dict(self, dict_n_d_rep):
        """
        Generates and adds graphs to self

        Args:
            dict_n_d_rep (dict): dict of dict of ints {n:{d:rep}} of graphs to generate
        """
        for n_key in dict_n_d_rep.keys():
            for d_key in dict_n_d_rep[n_key].keys():
                self.add_test_graphs(n_key, d_key, dict_n_d_rep[n_key][d_key])

    def add_test_graphs_with_list(self, n_d_repeats):
        """
        Generates and adds graphs to self

        Args:
            n_d_repeats (list): list of (n,d,repeats) tupples for graphs to be generated from
        """
        for n_d_rep in n_d_repeats:
            self.add_test_graphs(n_d_rep[0], n_d_rep[1], n_d_rep[2])

    def add_test_graphs(self, n, d, repeats):
        """
        Generates and adds "repeats" number of graphs with n nodes and d average connectivity
        to self

        Args:
            n (int): number of nodes in graphs to generate
            d (int): average connectivity of nodes in graphs to generate
            repeats (int): number of graphs to generate with parameters n, d
        """
        n = self.__check_if_pos_int(n, "n")
        d = self.__check_if_pos_int(d, "d")
        repeats = self.__check_if_pos_int(repeats, "repeats")
        if n not in self._graphs.keys():
            self._graphs[n] = {}
        if d not in self._graphs[n].keys():
            self._graphs[n][d] = []

        if n in self._known_graphs.keys():
            self.__get_known_graphs(n, d, repeats)
        else:
            for i in range(repeats):
                try:
                    self._graphs[n][d].append(self.create_3_colourable_graph(n, d))
                except RuntimeError as e:
                    print(repr(e))

    def get_graphs_all(self):
        """
        Returns all graphs saved in generator

        Returns:
            self._graphs (dict): dictionary of all graphs generated/requested
        """
        return self._graphs

    def get_graphs(self, n):
        """
        Returns all graphs with n nodes saved in generator

        Args:
            n (int): number of nodes of returned graphs

        Returns:
            dict: dictionary of all graphs with n nodes and d average connectivity
                generated/requested
        """
        return self._graphs[int(n)]

    def get_graphs(self, n, d):
        """
        Returns all graphs with n nodes and d average connectivity saved in generator

        Args:
            n (int): number of nodes of returned graphs
            d (int): average connectivity of returned graphs

        Returns:
            list: list of all graphs with n nodes and d average connectivity generated/requested
        """
        return self._graphs[int(n)][int(d)]

    def save(self):
        np.save(self._file_name, self._graphs)

    @staticmethod
    def create_3_colourable_graph(n, d):
        """
        Generates a random 3-colourable graph with average connectivity d

        Args:
            n (int): number of nodes in desired graph
            d (int): average connectivity of desired graph

        Returns:
            networkx.Graph: 3-colourable graph, n nodes d average connectivity
        """

        if d <= 0 or d >= n * 2 / 3:
            raise RuntimeError(f"d={d} is outside appropriate bounds")

        deletions = 0
        if n % 3 != 0:
            deletions = 3 - n % 3
            n += deletions
        d1 = int(d / 2)
        d2 = d1
        if d % 2 != 0:
            d2 += 1
        searching = True
        while searching:
            arr = np.arange(n, dtype=int)
            rng.shuffle(arr)

            V1, V2, V3 = np.split(arr, 3)

            graph = nx.Graph()
            graph.add_nodes_from(range(n))

            for Vr, Vs, d_rs in [(V1, V2, d1), (V1, V3, d2), (V2, V3, d2)]:
                bipartite_selected = False

                while not bipartite_selected:
                    edges = find_bipartite(Vr, Vs, d_rs)
                    if edges:
                        graph.add_edges_from(edges)
                        bipartite_selected = True

            if nx.is_connected(graph) and contains_triangle(graph):
                if deletions != 0:
                    graph.remove_nodes_from(range(n - deletions, n))
                    while not nx.is_connected(graph):
                        comps = list(nx.connected_components(graph))
                        graph.add_edge(list(comps[0])[0], list(comps[1])[0])
                return graph
            else:
                pass  # failed to find graph, retrying

    """Private"""

    def __get_known_graphs(self, n, d, repeats):
        graph_location = os.path.join(
            checked_files_location, f"number_of_nodes_{n}", f"connectivity_{d}.g6"
        )
        start = self._known_graphs[n][d]

        with open(graph_location, "rb") as f:
            num_lines = sum(1 for _ in f)
            if num_lines < (start + repeats):
                message = f"Not enough non-isomorphic graphs for n={n} d={d}"
                raise ValueError(message)

        file = open(graph_location, "rb")

        for i, line in enumerate(file):
            if i >= start and i < (start + repeats):
                self._known_graphs[n][d] += 1
                self._graphs[n][d].append(nx.from_graph6_bytes(line[:-1]))

    @staticmethod
    def __check_if_pos_int(num, name):
        try:
            num_float = float(num)
        except:
            raise ValueError(f"{name} {num} is not convertable to a number")
        if not num_float.is_integer():
            raise ValueError(f"{name} {num} is not an integer value")
        if num <= 0:
            raise ValueError(f"{name} {num} is not greater than 0")
        return num
