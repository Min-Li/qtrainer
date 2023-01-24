import pennylane as qml
from pennylane import numpy as qnp
import numpy as np


class BaseCircuit:

    def __init__(self, n_qubits: int, seed: int = None):
        self.n_qubits = n_qubits
        # self.params = self.initialize_params()
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            qnp.random.seed(seed)

    def set_params(self, params):
        if isinstance(params, tuple) or isinstance(params, list):
            self.params = [qnp.array(p, requires_grad=True) for p in params]
        elif isinstance(params, qnp.ndarray):
            self.params = params
        elif isinstance(params, np.ndarray):
            self.params = qnp.array(params, requires_grad=True)
        else:
            raise NotImplementedError(f'params type {type(params)} is not implemented.')

    def parameters(self):
        return self.params

    def initialize_params(self):
        pass

    def build_circuit(self):
        pass

    def cost_fn(self, *params):
        pass

    def sample_measurement(self, *params):
        pass

    def prob_measurement(self, *params):
        pass

    def grad(self, *params):
        return qml.grad(self.cost_fn)(*params)

    def parameter_shit(self, *params):
        return qml.jacobian(self.grad)(*params)
