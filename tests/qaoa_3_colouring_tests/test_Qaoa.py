import pytest
import pennylane as qml
from pennylane import numpy as qnp
import networkx as nx
from scipy.linalg import expm
import functools as ft
from qtcorgi import QubitQaoa
from qtcorgi import QubitSeparatedParameterQaoa
from qtcorgi import QutritQaoa

graph = nx.Graph([[0, 1], [0, 2], [1, 2]])
pauliZ = qnp.array([[1, 0], [0, -1]])
pauliZs = qnp.kron(pauliZ, pauliZ)
I = qnp.eye(2)
pauliZ_I = qnp.kron(pauliZ, I)
I_pauliZ = qnp.kron(I, pauliZ)


def valid_three_colouring(colouring):
    for node in graph.nodes:
        for neighbor in graph.neighbors(node):
            if colouring[node] == colouring[neighbor]:
                return False
    return "3" not in colouring


class TestQubitQaoa:
    def test_unitary(self):
        alpha = 2
        bit_test = QubitQaoa(graph=graph, fourth_colour_cost=alpha)
        gammas = [1, 0.6]
        betas = [0.4, 0.3]

        unitary = qml.matrix(bit_test.unitary_circuit)(gammas=gammas, betas=betas, n_layers=2)

        layer_Hadamard = self.get_Hadamards_unitary_matrix_of_graph(graph)
        layer_1 = self.get_unitary_matrix_of_graph(graph, gammas[0], betas[0], alpha)
        layer_2 = self.get_unitary_matrix_of_graph(graph, gammas[1], betas[1], alpha)

        expected_unitary = layer_2 @ layer_1 @ layer_Hadamard
        qnp.isclose(unitary, expected_unitary)
        assert qnp.allclose(unitary, expected_unitary)

    @classmethod
    def get_unitary_matrix_of_graph(cls, graph, gamma, beta, alpha):
        cost_matrix = cls.get_colour_cost_unitary_matrix_of_graph(graph, gamma, alpha)
        mixer_matrix = cls.get_mixer_unitary_matrix_of_graph(graph, beta)
        return mixer_matrix @ cost_matrix

    @staticmethod
    def get_colour_cost_unitary_matrix_of_graph(graph, gamma, alpha):
        H = TestQubitQaoa.get_colouring_Hamiltonian(graph)
        H += TestQubitQaoa.get_suppression_Hamiltonian(graph, alpha)

        return expm(-1j * gamma * H)

    @staticmethod
    def get_colouring_Hamiltonian(graph):
        num_qubits = 2 * graph.number_of_nodes()
        H = qnp.zeros((2**num_qubits, 2**num_qubits))
        for edge in graph.edges:
            qubits1 = [edge[0] * 2, (edge[0] * 2) + 1]
            qubits2 = [edge[1] * 2, (edge[1] * 2) + 1]

            I_before = qnp.eye(2 ** qubits1[0])
            I_between = qnp.eye(2 ** (qubits2[0] - qubits1[1] - 1))
            I_after = qnp.eye(2 ** (num_qubits - qubits2[1] - 1))

            H_1 = ft.reduce(qnp.kron, (I_before, pauliZs, I_between, pauliZs, I_after))
            H_2 = ft.reduce(qnp.kron, (I_before, pauliZ_I, I_between, pauliZ_I, I_after))
            H_3 = ft.reduce(qnp.kron, (I_before, I_pauliZ, I_between, I_pauliZ, I_after))
            H += H_1 + H_2 + H_3

        return H

    @staticmethod
    def get_suppression_Hamiltonian(graph, alpha):
        num_qubits = 2 * graph.number_of_nodes()
        H = qnp.zeros((2**num_qubits, 2**num_qubits))

        for node in graph.nodes:
            qubits = [node * 2, (node * 2) + 1]
            I_before = qnp.eye(2 ** qubits[0])
            I_after = qnp.eye(2 ** (num_qubits - qubits[1] - 1))

            H_1 = ft.reduce(qnp.kron, (I_before, pauliZs, I_after))
            H_2 = ft.reduce(qnp.kron, (I_before, pauliZ_I, I_after))
            H_3 = ft.reduce(qnp.kron, (I_before, I_pauliZ, I_after))
            H += H_1 - H_2 - H_3
        return alpha * H

    @staticmethod
    def get_mixer_unitary_matrix_of_graph(graph, beta):
        n = graph.number_of_nodes()
        RX = qml.RX.compute_matrix(beta)
        RX_per_node = [RX for _ in range(n * 2)]
        return ft.reduce(qnp.kron, RX_per_node)

    @staticmethod
    def get_Hadamards_unitary_matrix_of_graph(graph):
        n = graph.number_of_nodes()
        Hadamard = qml.Hadamard.compute_matrix()
        Hadamard_per_node = [Hadamard for _ in range(n * 2)]
        return ft.reduce(qnp.kron, Hadamard_per_node)

    def test_most_frequent_solution(self):
        bit_test = QubitSeparatedParameterQaoa(graph=graph, steps=25)
        most_frequent_quad_string = bit_test.qaoa_3_colouring(n_layers=2)[3]
        assert valid_three_colouring(most_frequent_quad_string)

    def test_ints_equiv_strings(self):
        num_samples = 200
        bit_test = QubitQaoa(graph=graph, samples=num_samples, learning_rate=1, steps=2)
        result_bit_test = bit_test.qaoa_3_colouring(n_layers=2)
        quad_string_ints = result_bit_test[1]
        quad_strings = result_bit_test[2]

        n = graph.number_of_nodes()
        quad_strings_from_ints = QubitQaoa.get_quad_strings_from_ints(quad_string_ints, n)[0]
        assert len(quad_strings) == num_samples
        assert len(quad_string_ints) == len(quad_strings)
        for i in range(num_samples):
            quad_string = "".join(quad_strings[i])
            assert quad_string == quad_strings_from_ints[i]

    def test_objectives_are_minimizing(self):
        bit_test = QubitQaoa(graph=graph, steps=10)
        results_low_steps = bit_test.qaoa_3_colouring(n_layers=2)

        bit_test.steps = 50
        results_high_steps = bit_test.qaoa_3_colouring(n_layers=2)

        assert results_high_steps[0] < results_low_steps[0], "Objective did not decrease"
        assert (
            results_high_steps[5] < results_low_steps[5]
        ), "Number sampled bad colourings did not decrease"


class TestQubitSeparatedParamQaoa:
    def test_unitary(self):
        alpha = 2
        bit_test = QubitSeparatedParameterQaoa(graph=graph, fourth_colour_cost=alpha)
        gammas = [1, 0.6]
        phis = [1.2, 0.7]
        betas = [0.4, 0.3]

        unitary = qml.matrix(bit_test.unitary_circuit)(
            gammas=gammas, phis=phis, betas=betas, n_layers=2
        )

        layer_Hadamard = TestQubitQaoa.get_Hadamards_unitary_matrix_of_graph(graph)
        layer_1 = self.get_unitary_matrix_of_graph(graph, gammas[0], phis[0], betas[0], alpha)
        layer_2 = self.get_unitary_matrix_of_graph(graph, gammas[1], phis[1], betas[1], alpha)

        expected_unitary = layer_2 @ layer_1 @ layer_Hadamard
        qnp.isclose(unitary, expected_unitary)
        assert qnp.allclose(unitary, expected_unitary)

    def test_most_frequent_solution(self):
        bit_test = QubitSeparatedParameterQaoa(graph=graph, steps=25)
        most_frequent_quad_string = bit_test.qaoa_3_colouring(n_layers=2)[3]
        assert valid_three_colouring(most_frequent_quad_string)

    def test_ints_equiv_strings(self):
        num_samples = 200
        bit_test = QubitSeparatedParameterQaoa(
            graph=graph, samples=num_samples, learning_rate=1, steps=2
        )
        result_bit_test = bit_test.qaoa_3_colouring(n_layers=2)
        quad_string_ints = result_bit_test[1]
        quad_strings = result_bit_test[2]

        n = graph.number_of_nodes()
        quad_strings_from_ints = QubitQaoa.get_quad_strings_from_ints(quad_string_ints, n)[0]
        assert len(quad_strings) == num_samples
        assert len(quad_string_ints) == len(quad_strings)
        for i in range(num_samples):
            quad_string = "".join(quad_strings[i])
            assert quad_string == quad_strings_from_ints[i]

    def test_objectives_are_minimizing(self):
        bit_test = QubitSeparatedParameterQaoa(graph=graph, steps=10)
        results_low_steps = bit_test.qaoa_3_colouring(n_layers=2)

        bit_test.steps = 50
        results_high_steps = bit_test.qaoa_3_colouring(n_layers=2)

        assert results_high_steps[0] < results_low_steps[0], "Objective did not decrease"
        assert (
            results_high_steps[5] < results_low_steps[5]
        ), "Number sampled bad colourings did not decrease"

    @classmethod
    def get_unitary_matrix_of_graph(cls, graph, gamma, phi, beta, alpha):
        cost_matrix = cls.get_cost_unitary_matrix_of_graph(graph, gamma, phi, alpha)
        mixer_matrix = TestQubitQaoa.get_mixer_unitary_matrix_of_graph(graph, beta)
        return mixer_matrix @ cost_matrix

    @classmethod
    def get_cost_unitary_matrix_of_graph(cls, graph, gamma, phi, alpha):
        colour_cost_matrix = cls.get_colour_cost_unitary_matrix_of_graph(graph, gamma)
        suppression_cost_matrix = cls.get_supression_cost_unitary_matrix_of_graph(graph, phi, alpha)
        return suppression_cost_matrix @ colour_cost_matrix

    @staticmethod
    def get_colour_cost_unitary_matrix_of_graph(graph, gamma):
        H = TestQubitQaoa.get_colouring_Hamiltonian(graph)
        return expm(-1j * gamma * H)

    @staticmethod
    def get_supression_cost_unitary_matrix_of_graph(graph, phi, alpha):
        H = TestQubitQaoa.get_suppression_Hamiltonian(graph, alpha)
        return expm(-1j * phi * H)


class TestQutritQaoa:
    @pytest.mark.skip(reason="qml.matrix of qutrit is broken, issue #4367")  # TODO
    def test_unitary(self):
        trit_test = QutritQaoa(graph=graph)
        gammas = [1, 0.6]
        betas = [0.4, 0.3]

        unitary = trit_test.unitary_circuit(gammas=gammas, betas=betas, n_layers=2)

        layer_1 = self.get_unitary_matrix_of_graph(graph, gammas[0], betas[0])
        layer_2 = self.get_unitary_matrix_of_graph(graph, gammas[1], betas[1])

        calculated_unitary = layer_2 @ layer_1
        assert qnp.allclose(qml.matrix(unitary), calculated_unitary)

    @classmethod
    def get_unitary_matrix_of_graph(cls, graph, gamma, beta):
        pass  # TODO add once PennyLane #4367 is fixed

    def test_most_frequent_solution(self):
        trit_test = QutritQaoa(graph=graph, steps=25)
        most_frequent_trit_string = trit_test.qaoa_3_colouring(n_layers=2)[3]
        assert valid_three_colouring(most_frequent_trit_string)

    def test_ints_equiv_strings(self):
        num_samples = 200
        trit_test = QutritQaoa(graph=graph, samples=num_samples, learning_rate=1, steps=2)
        edge_trit_test = trit_test.qaoa_3_colouring(n_layers=2)
        trit_string_ints = edge_trit_test[1]
        trit_strings = edge_trit_test[2]

        n = graph.number_of_nodes()
        trit_strings_from_ints = QutritQaoa.get_trit_strings_from_ints(trit_string_ints, n)
        assert len(trit_strings) == num_samples
        assert len(trit_string_ints) == len(trit_strings)
        for i in range(num_samples):
            assert trit_strings[i] == trit_strings_from_ints[i]

    def test_objective_is_minimizing(self):
        graph = nx.Graph([[0, 1], [0, 2], [1, 2]])

        trit_test = QutritQaoa(graph=graph, steps=2)
        objective_low_steps = trit_test.qaoa_3_colouring(n_layers=2)[0]

        trit_test.steps = 10
        objective_high_steps = trit_test.qaoa_3_colouring(n_layers=2)[0]

        assert objective_high_steps < objective_low_steps
