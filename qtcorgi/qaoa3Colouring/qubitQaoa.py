from pennylane import numpy as np
import pennylane as qml
import jax
from jax import numpy as jnp
import optax
from tqdm import tqdm


class QubitQaoa:
    """
    Implements qubit based QAOA method to solve 3-colouring problem on a graph.

    Args:
        graph (networkx.Graph): graph to be solved
        fourth_colour_cost (float): cost for fourth colour for qubit cost Hamiltonian
        samples (int): number of samples taken for solution
        learning_rate (float): optimization learning rate
        convergence_value (float): convergence value for qubit optimization halting
        steps (int): maximal number of optimization steps for qubits
        min_qubit_steps (int): minimum number of optimization steps done for qubit optimization
        print_position (int): position of loading bar
    """

    def __init__(
        self,
        graph,
        fourth_colour_cost=2,
        samples=1000,
        learning_rate=0.01,
        convergence_value=0.001,
        steps=1000,
        min_qubit_steps=200,
        print_position=0,
    ):
        self.graph = graph
        self.n_wires = graph.number_of_nodes() * 2
        self.H = self._get_ham(graph, fourth_colour_cost)

        self.fourth_colour_cost = fourth_colour_cost
        self.samples = samples
        self.convergence_value = convergence_value
        self.learning_rate = learning_rate
        self.steps = steps
        self.min_qubit_steps = min_qubit_steps
        self.print_position = print_position
        jax.config.update("jax_enable_x64", True)

    """Public Methods"""

    def qaoa_3_colouring(self, n_layers=3):
        """
        Qubit based algorithm for solving 3-colouring problem using qubit QAOA

        Args:
            n_layers (int): number of QAOA layers to use.

        Returns:
            objective_value (float): final expectation value of objective
            bit_string_ints (list): list of integer representations of bit-strings sampled
                after training
            quad_strings (list): list of sampled bit-strings concatenated into quad-strings
            most_freq_quad_string (str): most frequently sampled quad-string
            params (array): optimized parameters
            num_bad_strings (int): number of improperly coloured solutions sampled
        """

        # minimize the negative of the objective function
        def objective(params):
            return self._do_evolution_circuit(params, n_layers)

        init_params = 0.01 * np.random.rand(2, n_layers)
        return self._qaoa_qubit_3_colouring_optimization(n_layers, objective, init_params)

    def unitary_circuit(self, gammas, betas, n_layers):
        """
        Creates the circuit to be run for qubit QAOA algorithm

        Args:
            gammas (array): parameters for cost unitaries
            betas (array): parameters for mixer unitaries
            n_layers (int): number of layers
        """
        for wire in range(self.n_wires):
            qml.Hadamard(wires=wire)
        for layer in range(n_layers):
            self._enact_unitary_cost(self.graph, gammas[layer], self.fourth_colour_cost)
            self._enact_unitary_mixer(self.graph, betas[layer])

    @classmethod
    def get_quad_strings_from_ints(cls, bit_string_ints, num_nodes):
        """
        Get equivalent quad-strings from a list of integers

        Args:
            bit_string_ints (list): list of integers corresponding to sampled strings
            num_nodes (int): number of nodes in graph

        Returns:
            quad_strings (list): list of quad-strings colouring graph
            num_bad_strings (int): number of strings including the fourth colour
        """
        quad_strings = []
        num_bad_strings = 0
        for bs_int in bit_string_ints:
            quad_string, num_bad_strings = cls._get_quad_string_from_int(
                bs_int, num_nodes, num_bad_strings
            )
            quad_strings.append(quad_string)
        return quad_strings, num_bad_strings

    """Private Methods"""

    def _qaoa_qubit_3_colouring_optimization(self, n_layers, objective, init_params):
        # initialize the parameters near zero
        params = jnp.array(init_params)

        optimizer = optax.adam(self.learning_rate)
        opt_state = optimizer.init(params)

        obj_fn = jax.jit(objective)
        grad_fn = jax.jit(jax.grad(obj_fn))
        opt_update = jax.jit(optimizer.update)
        apply_updates = jax.jit(optax.apply_updates)

        last_obj = np.inf
        laststep_conv = False

        iterator = tqdm(
            range(self.steps),
            position=self.print_position,
            desc="Qubit: Solving graph instance (will stop early)",
            leave=False,
        )
        for i in iterator:
            grads = grad_fn(params)
            updates, opt_state = opt_update(grads, opt_state)
            params = apply_updates(params, updates)
            new_obj = obj_fn(params)

            if i >= self.min_qubit_steps and np.abs(new_obj - last_obj) < self.convergence_value:
                if laststep_conv:
                    iterator.close()
                    break
                laststep_conv = True
            else:
                laststep_conv = False
            last_obj = new_obj

        bit_strings = self._do_samples_circuit(params, n_layers)
        bit_string_ints = self._bit_string_to_int(bit_strings)
        quad_strings, num_bad_strings = self.get_quad_strings_from_ints(
            bit_string_ints, self.graph.number_of_nodes()
        )

        # get optimal parameters and most frequently sampled bitstring
        counts = np.bincount(np.array(bit_string_ints))
        most_freq_quad_string, _ = self._get_quad_string_from_int(
            np.argmax(counts), self.graph.number_of_nodes(), 0
        )

        return (
            obj_fn(params),
            bit_string_ints,
            quad_strings,
            most_freq_quad_string,
            params,
            num_bad_strings,
        )

    def _do_samples_circuit(self, params, n_layers):
        @qml.qnode(
            qml.device("default.qubit", wires=self.n_wires, shots=self.samples),
            interface="jax",
        )
        def circuit(gammas, betas):
            self.unitary_circuit(gammas, betas, n_layers)
            return qml.sample()

        return circuit(params[0], params[1])

    def _do_evolution_circuit(self, params, n_layers):
        @qml.qnode(qml.device("default.qubit", wires=self.n_wires), interface="jax")
        def circuit(gammas, betas):
            self.unitary_circuit(gammas, betas, n_layers)
            return qml.expval(self.H)

        return circuit(params[0], params[1])

    def _bit_string_to_int(self, bit_string_sample):
        weights = np.arange(np.shape(bit_string_sample)[1] - 1, -1, -1, dtype=int)
        weights = 2**weights
        return np.sum(bit_string_sample * weights, axis=1)

    def _get_qubits_of_node(self, node):
        qubit1 = node * 2
        return [qubit1, qubit1 + 1]

    def _enact_unitary_mixer(self, graph, beta):
        for node in graph.nodes:
            qubits = self._get_qubits_of_node(node)
            qml.RX(beta, wires=qubits[0])
            qml.RX(beta, wires=qubits[1])

    def _enact_unitary_cost(self, graph, gamma, fourth_colour_cost):
        gamma *= 2
        for edge in graph.edges:
            qubits1 = self._get_qubits_of_node(edge[0])
            qubits2 = self._get_qubits_of_node(edge[1])
            qml.CNOT(wires=[qubits1[0], qubits2[0]])
            qml.CNOT(wires=[qubits1[1], qubits2[1]])
            qml.CNOT(wires=[qubits2[0], qubits2[1]])
            qml.RZ(gamma, wires=qubits2[1])

            qml.RZ(gamma, wires=qubits2[0])
            qml.CNOT(wires=[qubits2[0], qubits2[1]])
            qml.CNOT(wires=[qubits1[0], qubits2[0]])

            qml.RZ(gamma, wires=qubits2[1])
            qml.CNOT(wires=[qubits1[1], qubits2[1]])

        for node in graph.nodes:
            qubits = self._get_qubits_of_node(node)

            qml.CNOT(wires=qubits)
            qml.RZ(fourth_colour_cost * gamma, wires=[qubits[1]])
            qml.CNOT(wires=qubits)
            qml.RZ(-fourth_colour_cost * gamma, wires=qubits[0])
            qml.RZ(-fourth_colour_cost * gamma, wires=qubits[1])

    @staticmethod
    def _get_quad_string_from_int(bs_int, num_nodes, num_bad_strings):
        quad_string = np.base_repr(bs_int, base=4)
        if "3" in quad_string:
            num_bad_strings += 1
        leading_zeros = "0" * (num_nodes - len(quad_string))
        return leading_zeros + quad_string, num_bad_strings

    def _get_ham(self, graph, fourth_colour_cost):
        Hs = []
        Hparams = []
        for edge in graph.edges:
            qubits1 = self._get_qubits_of_node(edge[0])
            qubits2 = self._get_qubits_of_node(edge[1])

            Hs.append(
                qml.PauliZ(wires=qubits1[0])
                @ qml.PauliZ(wires=qubits1[1])
                @ qml.PauliZ(wires=qubits2[0])
                @ qml.PauliZ(wires=qubits2[1])
            )
            Hs.append(qml.PauliZ(wires=qubits1[0]) @ qml.PauliZ(wires=qubits2[0]))
            Hs.append(qml.PauliZ(wires=qubits1[1]) @ qml.PauliZ(wires=qubits2[1]))
            Hparams += [1, 1, 1]

        for node in graph.nodes:
            qubits = self._get_qubits_of_node(node)
            Hs.append(qml.PauliZ(wires=qubits[0]) @ qml.PauliZ(wires=qubits[1]))
            Hs.append(qml.PauliZ(wires=qubits[0]))
            Hs.append(qml.PauliZ(wires=qubits[1]))
            Hparams += [fourth_colour_cost, -fourth_colour_cost, -fourth_colour_cost]

        return qml.Hamiltonian(Hparams, Hs)


class QubitSeparatedParameterQaoa(QubitQaoa):
    """
    Implements qubit based algorithm method to solve 3-colouring problem on a graph.
    The algorithm is based on QAOA however the two parts, colouring-cost and
        fourth-colour-suppression-cost, of the cost unitary have separate trainable parameters.

    Args:
        graph (networkx.Graph): graph to be solved
        fourth_colour_cost (float): cost for fourth colour for qubit cost Hamiltonian
        samples (int): number of samples taken for solution
        learning_rate (float): optimization learning rate
        convergence_value (float): convergence value for qubit optimization halting
        steps (int): maximal number of optimization steps for qubits
        min_qubit_steps (int): minimum number of optimization steps done for qubit optimization
        print_position (int): position of loading bar
    """

    def qaoa_3_colouring(self, n_layers=3):
        """
        Qubit based algorithm for solving 3-colouring problem using qubit algorithm
            colouring-cost and fourth-colour-suppression-cost parameters separated

        Args:
            n_layers (int): number of QAOA layers to use.

        Returns:
            objective_value (float): final expectation value of objective
            bit_string_ints (list): list of integer representations of bit-strings sampled
                after training
            quad_strings (list): list of sampled bit-strings concatenated into quad-strings
            most_freq_quad_string (str): most frequently sampled quad-string
            params (array): optimized parameters
            num_bad_strings (int): number of improperly coloured solutions sampled
        """

        # minimize the negative of the objective function
        def objective(params):
            return self._do_evolution_circuit(params, n_layers)

        init_params = 0.01 * np.random.rand(3, n_layers)
        return self._qaoa_qubit_3_colouring_optimization(n_layers, objective, init_params)

    def unitary_circuit(self, gammas, phis, betas, n_layers):
        """
        Creates the circuit to be run for qubit QAOA algorithm with separated colouring and
        suppression cost parameters

        Args:
            gammas (array): parameters for colouring cost unitaries
            phis (array): parameters for fourth colour suppression cost unitaries
            betas (array): parameters for mixer unitaries
            n_layers (int): number of layers
        """
        for wire in range(self.n_wires):
            qml.Hadamard(wires=wire)
        for layer in range(n_layers):
            self._enact_unitary_cost(
                self.graph, gammas[layer], phis[layer], self.fourth_colour_cost
            )
            self._enact_unitary_mixer(self.graph, betas[layer])

    """Private"""

    def _enact_unitary_cost(self, graph, gamma, phi, fourth_colour_cost):
        gamma *= 2
        phi *= 2
        for edge in graph.edges:
            qubits1 = self._get_qubits_of_node(edge[0])
            qubits2 = self._get_qubits_of_node(edge[1])
            qml.CNOT(wires=[qubits1[0], qubits2[0]])
            qml.CNOT(wires=[qubits1[1], qubits2[1]])
            qml.CNOT(wires=[qubits2[0], qubits2[1]])
            qml.RZ(gamma, wires=qubits2[1])

            qml.RZ(gamma, wires=qubits2[0])
            qml.CNOT(wires=[qubits2[0], qubits2[1]])
            qml.CNOT(wires=[qubits1[0], qubits2[0]])

            qml.RZ(gamma, wires=qubits2[1])
            qml.CNOT(wires=[qubits1[1], qubits2[1]])

        for node in graph.nodes:
            qubits = self._get_qubits_of_node(node)

            qml.CNOT(wires=qubits)
            qml.RZ(fourth_colour_cost * phi, wires=[qubits[1]])
            qml.CNOT(wires=qubits)
            qml.RZ(-fourth_colour_cost * phi, wires=qubits[0])
            qml.RZ(-fourth_colour_cost * phi, wires=qubits[1])

    def _do_samples_circuit(self, params, n_layers):
        gammas = params[0]
        phis = params[1]
        betas = params[2]

        @qml.qnode(
            qml.device("default.qubit", wires=self.n_wires, shots=self.samples),
            interface="jax",
        )
        def circuit(gammas, phis, betas):
            self.unitary_circuit(gammas, phis, betas, n_layers)
            return qml.sample()

        return circuit(gammas, phis, betas)

    def _do_evolution_circuit(self, params, n_layers):
        gammas = params[0]
        phis = params[1]
        betas = params[2]

        @qml.qnode(qml.device("default.qubit", wires=self.n_wires), interface="jax")
        def circuit(gammas, phis, betas):
            self.unitary_circuit(gammas, phis, betas, n_layers)
            return qml.expval(self.H)

        return circuit(gammas, phis, betas)
