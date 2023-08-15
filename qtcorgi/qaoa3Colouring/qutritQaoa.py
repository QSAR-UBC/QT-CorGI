from pennylane import numpy as np
import pennylane as qml
import jax
from jax import numpy as jnp
import optax
from tqdm import tqdm

TSub = qml.adjoint(qml.TAdd)


class QutritQaoa:
    """
    Implements qutrit based QAOA method to solve 3-colouring problem on a graph.

    Args:
        graph (networkx.Graph): graph to be solved
        samples (int): number of samples taken for solution
        learning_rate (float): optimization learning rate
        steps (int): number of optimization steps
        print_position (int): position of loading bar
    """

    def __init__(self, graph, samples=1000, learning_rate=0.01, steps=50, print_position=0):
        self.graph = graph
        self.n_wires = graph.number_of_nodes()
        self.H = self.__get_ham()
        self.samples = samples
        self.learning_rate = learning_rate
        self.steps = steps
        self.print_position = print_position
        jax.config.update("jax_enable_x64", True)

    """Public Methods"""

    def qaoa_3_colouring(self, n_layers=3):
        """
        Qutrit based algorithm for solving 3-colouring problem using qutrit QAOA

        Args:
            n_layers (int): number of QAOA layers to use.

        Returns:
            objective_value (float): final expectation value of objective
            trit_string_ints (list): list of integer representations of trit-strings sampled
                after training
            trit_strings (list): list of sampled trit-strings
            most_freq_trit_string (str): most frequently sampled trit-string
            params (array): optimized parameters
        """

        # minimize the cost function
        def objective(params):
            gammas = params[0]
            betas = params[1]
            return self.__do_evolution_circuit(gammas, betas, n_layers=n_layers)

        return self.__qaoa_qutrit_3_colouring(n_layers, objective)

    def unitary_circuit(self, gammas, betas, n_layers):
        """
        Creates the circuit to be run for qutrit QAOA algorithm

        Args:
            gammas (array): parameters for cost unitaries
            betas (array): parameters for mixer unitaries
            n_layers (int): number of layers
        """
        for wire in range(self.n_wires):
            qml.THadamard(wires=wire)

        # p instances of unitary operators
        for i in range(n_layers):
            # cost
            for edge in self.graph.edges:
                wire1 = edge[0]
                wire2 = edge[1]

                TSub(wires=[wire1, wire2])
                qml.TRZ((4 / 3) * gammas[i], wires=wire2, subspace=[0, 1])
                qml.TRZ((4 / 3) * gammas[i], wires=wire2, subspace=[0, 2])
                qml.TAdd(wires=[wire1, wire2])

            for node in self.graph.nodes:
                qml.TRX(betas[i], wires=node, subspace=[0, 1])
                qml.TRX(betas[i], wires=node, subspace=[1, 2])
                qml.TRX(betas[i], wires=node, subspace=[0, 2])

    @classmethod
    def get_trit_strings_from_ints(cls, trit_string_ints, num_nodes):
        """
        Get equivalent trit-strings from a list of integers

        Args:
            trit_string_ints (list): list of integers corresponding to sampled strings
            num_nodes (int): number of nodes in graph

        Returns:
            trit_strings (list): list of trit-strings colouring graph
        """
        trit_strings = []
        for ts_int in trit_string_ints:
            trit_string = cls._get_trit_string_from_int(ts_int, num_nodes)
            trit_strings.append(trit_string)
        return trit_strings

    @staticmethod
    def _get_trit_string_from_int(ts_int, num_nodes):
        trit_string = np.base_repr(ts_int, base=3)
        leading_zeros = "0" * (num_nodes - len(trit_string))
        return leading_zeros + trit_string

    """Private Methods"""

    def __trit_string_to_int(self, trit_string_sample):
        weights = np.arange(np.shape(trit_string_sample)[1] - 1, -1, -1, dtype=int)
        weights = 3**weights
        return np.sum(trit_string_sample * weights, axis=1)

    def __get_gellManns_for_edge(self, edge):
        gell_mann_3_kroned = qml.GellMann(wires=edge[0], index=3) @ qml.GellMann(
            wires=edge[1], index=3
        )
        gell_mann_8_kroned = qml.GellMann(wires=edge[0], index=8) @ qml.GellMann(
            wires=edge[1], index=8
        )

        return [gell_mann_3_kroned, gell_mann_8_kroned]

    def __get_most_frequent_trit_string_ternary(self, trit_string_ints, n_wires):
        counts = np.bincount(trit_string_ints)
        most_freq_trit_string = np.base_repr(np.argmax(counts), base=3)
        return most_freq_trit_string.rjust(n_wires, "0")

    def __qaoa_qutrit_3_colouring(self, n_layers, objective):
        # initialize the parameters near zero
        init_params = 0.01 * np.random.rand(2, n_layers)
        params = jnp.array(init_params)

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        obj_fn = jax.jit(objective)
        grad_fn = jax.jit(jax.grad(obj_fn))
        opt_update = jax.jit(optimizer.update)
        apply_updates = jax.jit(optax.apply_updates)

        for _ in tqdm(
            range(self.steps),
            desc="Qutrit: Solving graph instance",
            leave=False,
            position=self.print_position,
        ):
            grads = grad_fn(params)
            updates, opt_state = opt_update(grads, opt_state)
            params = apply_updates(params, updates)

        trit_arrays = self.__do_samples_circuit(params[0], params[1], n_layers=n_layers)
        trit_string_ints = self.__trit_string_to_int(trit_arrays)

        get_trit_string = lambda trit_array: "".join(str(trit) for trit in trit_array)
        trit_strings = [get_trit_string(trit_array) for trit_array in trit_arrays]

        # Find optimal parameters and most frequently sampled bitstring
        most_freq_trit_string = self.__get_most_frequent_trit_string_ternary(
            trit_string_ints, self.n_wires
        )

        return (obj_fn(params), trit_string_ints, trit_strings, most_freq_trit_string, params)

    def __do_evolution_circuit(self, gammas, betas, n_layers=1):
        @qml.qnode(qml.device("default.qutrit", wires=self.n_wires), interface="jax")
        def circuit(gammas, betas):
            self.unitary_circuit(gammas, betas, n_layers)
            return qml.expval(self.H)

        return circuit(gammas, betas)

    def __do_samples_circuit(self, gammas, betas, n_layers=1):
        @qml.qnode(
            qml.device("default.qutrit", wires=self.n_wires, shots=self.samples),
            interface="jax",
        )
        def circuit(gammas, betas):
            self.unitary_circuit(gammas, betas, n_layers)

            return qml.sample()

        return circuit(gammas, betas)

    def __get_ham(self):
        ham_list = []
        for edge in self.graph.edges:
            ham_list += self.__get_gellManns_for_edge(edge)

        return qml.Hamiltonian(np.ones(len(ham_list)), ham_list)
