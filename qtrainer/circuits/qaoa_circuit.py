import pennylane as qml
from pennylane import qaoa
import networkx as nx
from pennylane import numpy as qnp  # qml's numpy
from .base_circuit import BaseCircuit


class QAOACircuit(BaseCircuit):
    QAOA_TASKS = ['bit_driver', 'edge_driver', 'maxcut', 'min_vertex_cover', 'max_clique', 'max_independent_set']

    def __init__(self, graph: nx.Graph, task: str, depth: int, seed: int = None,
                 # device='lightning.qubit',
                 **task_args):
        self.graph = graph

        cost_h, mixer_h = getattr(qaoa, task)(graph,
                                              **task_args)  # e.g., for maxcut, this = qaoa.maxcut(graph, constrained=task_args['constrained'])
        self.cost_h = cost_h
        self.mixer_h = mixer_h
        self.depth = depth

        super().__init__(n_qubits=graph.order(), seed=seed)
        self.initialize_params()

    def initialize_params(self):
        self.params = [2 * qnp.pi * (qnp.random.rand(self.depth, requires_grad=True) - 0.5),
                2 * qnp.pi * (qnp.random.rand(self.depth, requires_grad=True) - 0.5)]
        return self.params

    def build_circuit(self):
        pass

    def qaoa_layer_fn(self, gamma, alpha, ):
        qaoa.cost_layer(gamma, self.cost_h)
        qaoa.mixer_layer(alpha, self.mixer_h)

    def ansatz(self, *params, n_qubits: int, depth: int):
        # initialize all qubits into +X eigenstate.
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        gammas = params[0]
        alphas = params[1]
        # stack building blocks for depth times.
        qml.layer(self.qaoa_layer_fn, depth, gammas, alphas)

    def cost_fn(self, *params):
        self.ansatz(*params, n_qubits=self.n_qubits, depth=self.depth)
        return qml.expval(self.cost_h)

    def sample_measurement(self, *params):
        self.ansatz(*params, n_qubits=self.n_qubits, depth=self.depth)
        return [qml.sample(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def prob_measurement(self, *params):
        self.ansatz(*params, n_qubits=self.n_qubits, depth=self.depth)
        return qml.probs(wires=list(range(self.n_qubits)))
