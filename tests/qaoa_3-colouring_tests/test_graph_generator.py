import pytest
import networkx as nx
import numpy as np
from qtcorgi.qaoa3Colouring.find3ColourableGraphs import checkGraphs
from qtcorgi import GraphGenerator


class TestGraph_Checker:
    @pytest.mark.parametrize(
        "graph,is_three_colourable",
        [
            (nx.Graph([[0, 1]]), True),
            (nx.Graph([[0, 1], [1, 2], [2, 1]]), True),
            (nx.Graph([[0, 1], [1, 2], [2, 1], [2, 3], [1, 3], [0, 2]]), True),
            (
                nx.Graph([[0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
                True,
            ),  # Not solved by "greedy" check
            (nx.Graph([[0, 1], [1, 2], [2, 1], [2, 3], [1, 3], [0, 2], [0, 3], [3, 4]]), False),
            (nx.Graph([[0, 1], [1, 2], [2, 1], [2, 3], [1, 3], [0, 2], [0, 4], [3, 4]]), True),
        ],
    )
    def test_graph_three_colourable(self, graph, is_three_colourable):
        check_three_colourable = checkGraphs.is_graph_three_colourable(graph, np.inf)
        assert is_three_colourable == check_three_colourable

    @pytest.mark.parametrize(
        "num_nodes,true_max_edges",
        [
            (10, 33),
            (11, 40),
            (32, 341),
        ],
    )
    def test_max_three_colourable_edges(self, num_nodes, true_max_edges):
        max_edges = checkGraphs.get_max_edges(num_nodes)
        assert max_edges == true_max_edges


class TestGraphGeneration:
    @pytest.mark.parametrize(
        "input_dictionary",
        [
            {4: {2: 3}},
            {4: {2: 3}, 5: {3: 1}},
            {5: {2: 1, 3: 1}},
            {15: {3: 1}},
            {15: {3: 3, 9: 3}},
            {4: {2: 3}, 15: {3: 1}},
        ],
    )
    def test_add_graphs_dictionary(self, input_dictionary):
        errors = []
        graph_gen = GraphGenerator()
        graph_gen.add_test_graphs_with_dict(input_dictionary)

        graph_dict = graph_gen.get_graphs_all()

        for n in input_dictionary.keys():
            max_edges = checkGraphs.get_max_edges(n)
            for d in input_dictionary[n].keys():
                repeats = input_dictionary[n][d]
                graphs = graph_dict[n][d]
                if len(graphs) != repeats:
                    errors.append(f"{len(graphs)} graphs, not {repeats} for n={n}, d={d}")
                for i, graph in enumerate(graphs):
                    num_nodes = graph.number_of_nodes()
                    if num_nodes != n:
                        errors.append(f"For n={n}, d={d}: graph {i} has wrong number of nodes")
                    if not checkGraphs.is_graph_three_colourable(graph, max_edges):
                        errors.append(f"For n={n}, d={d}: graph {i} is not 3 colourable")
        assert not errors, "errors occured:\n{}".format("\n".join(errors))

    @pytest.mark.parametrize(
        "input_list",
        [
            [(4, 2, 3)],
            [(4, 2, 3), (5, 3, 1)],
            [(5, 2, 1), (5, 3, 1)],
            [(15, 3, 3), (15, 9, 3)],
            [(14, 3, 3), (15, 3, 3)],
            [(4, 2, 3), (15, 3, 1)],
        ],
    )
    def test_add_graphs_list(self, input_list):
        errors = []
        graph_gen = GraphGenerator()
        graph_gen.add_test_graphs_with_list(input_list)
        graph_dict = graph_gen.get_graphs_all()

        for n_d_rep in input_list:
            n = n_d_rep[0]
            d = n_d_rep[1]
            repeats = n_d_rep[2]

            graphs = graph_dict[n][d]
            if len(graphs) != repeats:
                errors.append(f"{len(graphs)} graphs, not {repeats} for n={n}, d={d}")
            for i, graph in enumerate(graphs):
                num_nodes = graph.number_of_nodes()
                if num_nodes != n:
                    errors.append(
                        f"For n={n}, d={d}: graph {i} has wrong number of nodes: {num_nodes}"
                    )
                # skipping check three-colourable.
        assert not errors, "errors occured:\n{}".format("\n".join(errors))

    @pytest.mark.parametrize("n,d", [(3, -1), (3, 0), (15, 10), (14, 10)])
    def test_add_bad_d(self, n, d):
        graph_gen = GraphGenerator()
        with pytest.raises(ValueError):
            graph_gen.add_test_graphs(n=4, d=2, repeats=6)

    def test_add_too_many_small_graphs(self):
        graph_gen = GraphGenerator()
        with pytest.raises(ValueError):
            graph_gen.add_test_graphs(n=4, d=2, repeats=6)

    def test_add_too_many_small_graphs_second_time(self):
        graph_gen = GraphGenerator()
        graph_gen.add_test_graphs(n=4, d=2, repeats=3)
        with pytest.raises(ValueError):
            graph_gen.add_test_graphs(n=4, d=2, repeats=3)
