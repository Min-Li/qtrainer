import pennylane as qml
from pennylane import qchem
from pennylane import qaoa
import networkx as nx
from pennylane import numpy as qnp  # qml's numpy
import numpy as np
from .base_circuit import BaseCircuit
from typing import List, Union, Dict, Tuple


class VQECircuit(BaseCircuit):

    def __init__(self, ansatz: callable, n_qubits: int, Hamiltonian: qml.Hamiltonian,
                 init_params: Union[List, Tuple],
                 ):
        # Hamiltonian, n_qubits = qchem.molecular_hamiltonian(symbols, coordinates, **task_args)

        # self.device = qml.device(device, wires=graph.order())
        super().__init__(n_qubits=n_qubits)
        self.ansatz = ansatz
        self.Hamiltonian = Hamiltonian
        self.n_qubits = n_qubits
        self.params = init_params
        self.init_params = init_params  # save as a backup


    def initialize_params(self):
        self.params = self.init_params
        return self.params

    def cost_fn(self, *params):
        self.ansatz(*params, )
        return qml.expval(self.Hamiltonian)

