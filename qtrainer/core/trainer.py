import pickle

import numpy as np
import pandas as pd
import pennylane as qml
import wandb
from tensorboardX import SummaryWriter
from tqdm.auto import trange

from qtrainer.circuits import BaseCircuit
from .optimizers import get_optimizer


class QTrainerLogger():
    def __init__(self, root: str = None, experiment_name: str = None, use_tensorboard=False,
                 dataframe_format="csv",
                 use_wandb=False, wandb_project=None, wandb_entity=None, wandb_group=None, ):
        self.params_history = {}
        self.cost_history = {}
        self.grad_history = {}
        self.writer = None
        self.log_dir = root + "/" + experiment_name if root is not None else None
        if use_tensorboard:
            assert self.log_dir is not None, "log_dir must be specified if use_tensorboard is True"
            self.writer = SummaryWriter(log_dir=self.log_dir)
        self.dataframe_format = dataframe_format

        self.use_wandb = use_wandb
        if use_wandb:
            assert self.log_dir is not None, "log_dir must be specified if use_wandb is True"
            wandb.init(project=wandb_project, entity=wandb_entity, dir=self.log_dir, name=experiment_name,
                       group=wandb_group)

    def log(self, step, params=None, cost=None, grad=None):
        if params is not None:
            params = [np.array(param) for param in params]
            self.params_history[step] = params
        if cost is not None:
            self.cost_history[step] = cost
            # Log to tensorboard
            if self.writer is not None:
                self.writer.add_scalar("cost", cost, step)

        if grad is not None:
            grad = [np.array(g) for g in grad]
            self.grad_history[step] = grad

    @property
    def optimal_params(self):
        # find the step of the minimum cost
        min_cost = min(self.cost_history.values())
        for step, cost in self.cost_history.items():
            if cost == min_cost:
                return self.params_history[step]

    def close(self):
        if self.writer is not None:
            self.writer.close()
        # Export dataframe
        self.export_dataframe(self.log_dir + "/train_history", format=self.dataframe_format)

        if self.log_dir is not None:
            # Save parameter history using Pickle
            pickle.dump(self.params_history, open(self.log_dir + "/params_history.pkl", "wb"))
            # Save gradient history using Pickle
            pickle.dump(self.grad_history, open(self.log_dir + "/grad_history.pkl", "wb"))
        print("QTrainerLogger closed - see logs in {}".format(self.log_dir))

    def export_dataframe(self, file_path: str, format="csv"):
        df = pd.DataFrame(columns=["step", "cost", ])
        for step, cost in self.cost_history.items():
            df = df.append(
                {
                    "step": step,
                    "cost": cost,
                },
                ignore_index=True
            )
        if format == "csv":
            df.to_csv(file_path, index=False)
        elif format == "json":
            df.to_json(file_path, orient="records")
        elif format == 'excel':
            df.to_excel(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format {format}")


class Trainer:
    def __init__(self, circuit: BaseCircuit,
                 optimizer: str = 'Adam',
                 n_steps: int = 100,
                 shots_per_step: int = None,
                 error_mitigation_method: str = None,
                 error_mitigation_budget: int = None,
                 error_mitigation_config: dict = None,
                 optimizer_config: dict = None,
                 device_name: str = None,
                 device_config: dict = None,
                 grad_method: str = 'best',  # method to compute gradient; default is autograd (qml.grad)'
                 eval_freq: int = 10,
                 model_select: str = 'best',
                 noise_fn: callable = None,
                 logger: str = None,
                 # **kwargs
                 ):
        self.circuit = circuit
        self.optimizer = optimizer

        self.n_steps = n_steps
        # self.kwargs = kwargs
        self.n_qubits = circuit.n_qubits
        self.device_name = device_name
        self.device_config = {} if device_config is None else device_config
        self.shots_per_step = shots_per_step
        # Construct Device
        self.noise_fn = noise_fn
        self.device = self.setup_device(backup_devices=True,shots=self.shots_per_step)
        self.optimizer_config = optimizer_config or {}
        self.optimizer = get_optimizer(optimizer, **self.optimizer_config)
        
        self.cost_fn = self.cost_fn_original = qml.QNode(circuit.cost_fn,
                                                         self.device,
                                                         diff_method=grad_method
                                                         )  # Run the circuit with the device
        self.grad_method = None  # ['param_shift', 'finite_diff']
        # Quantum Error Mitigation
        self.error_mitigation_method = error_mitigation_method
        self.error_mitigation_budget = error_mitigation_budget
        self.error_mitigation_config = error_mitigation_config or {}
        if self.error_mitigation_method is not None:
            # Error-mitigated cost function
            self.cost_fn = self.apply_error_mitigation(self.cost_fn)
        self.model_select = model_select  # ['best', 'last']
        self.grad_fn = self.get_grad_fn()
        self.eval_freq = eval_freq

        self.logger = QTrainerLogger(use_wandb=logger == "wandb", use_tensorboard=logger == "tensorboard",)

    def setup_device(self, shots: int = None, backup_devices: bool = False):
        device = qml.device(self.device_name, wires=self.circuit.n_qubits, shots=shots,**self.device_config)
        if self.noise_fn is None:
            return device
        else:
            device_noisy = self.noise_fn(device)
            if backup_devices:
                self.device_original = device
                self.device_noisy = device_noisy
            return device_noisy

    def apply_error_mitigation(self, circuit_fn: qml.QNode):
        if self.error_mitigation_method.lower() == 'zne':
            error_mitigated_circuit_fn = qml.transforms.mitigate_with_zne(
                circuit_fn,
                scale_factors=self.error_mitigation_config.get('scale_factors', [1, 2, 3]),
                folding=qml.transforms.fold_global,
                extrapolate=qml.transforms.richardson_extrapolate,
            )
        else:
            raise NotImplementedError(f'Error mitigation method {self.error_mitigation_method} is not implemented.')

        # self.cost_fn_original = self.cost_fn
        self.cost_fn_mitigated = error_mitigated_circuit_fn
        return error_mitigated_circuit_fn

    def prob_measurement(self, params=None):
        params = params or self.params
        # device = qml.device(self.device_name, wires=self.n_qubits, ) if self.device.shots is not None else self.device
        prob_fn = qml.QNode(self.circuit.prob_measurement, self.device, shots=None)
        return prob_fn(*params)

    def sample_measurement(self, shots: int, params=None, error_mitigate=False):
        params = params or self.params
        # device = qml.device(self.device_name, wires=self.n_qubits, shots=shots)
        device = self.setup_device(shots=shots)
        sample_fn = qml.QNode(self.circuit.sample_measurement, device)
        if error_mitigate:
            sample_fn = self.apply_error_mitigation(sample_fn)
        return sample_fn(*params)

    def get_grad_fn(self, ):
        if self.error_mitigation_method is None or self.error_mitigation_method.lower() == 'zne':
            if self.grad_method is None:
                grad_fn = qml.grad
            else:
                grad_fn = getattr(qml.gradients, self.grad_method)
        else:
            raise NotImplementedError
        return grad_fn

    def compute_gradient(self, params, cost_fn: callable = None):
        cost_fn = cost_fn or self.cost_fn
        grad = self.grad_fn(cost_fn)(*params)
        return grad

    def get_cost_fn(self, cost_fn: callable = None, cost_fn_version: str = None):
        assert not (cost_fn is not None and cost_fn_version is not None), 'cannot be both None'
        cost_fn = cost_fn or self.cost_fn
        if cost_fn_version is not None:
            if cost_fn_version == 'original':
                cost_fn = self.cost_fn_original
            elif cost_fn_version == 'mitigated':
                cost_fn = self.cost_fn_mitigated
            else:
                raise NotImplementedError(f'cost_fn_version {cost_fn_version} is not implemented.')
        return cost_fn

    def train_step(self, params=None, cost_fn: callable = None, cost_fn_version: str = None):
        params = params or self.params
        # n_trainable_tensors = sum(getattr(p, "requires_grad", False) for p in params)
        assert isinstance(params, tuple) or isinstance(params, list)
        cost_fn = self.get_cost_fn(cost_fn, cost_fn_version)
        grad = self.compute_gradient(params, cost_fn)
        new_params = self.optimizer.apply_grad(grad, params, )
        self.set_params(new_params)
        return new_params

    def eval_cost(self, params=None, cost_fn: callable = None, cost_fn_version: str = None):
        params = params or self.params
        cost_fn = self.get_cost_fn(cost_fn, cost_fn_version)
        return cost_fn(*params).mean()

    def draw_circuit(self, params, return_fig=False):
        if return_fig:
            fig, ax = qml.draw_mpl(self.cost_fn)(params)
            return fig
        else:
            return qml.draw(self.cost_fn)(params)

    @property
    def params(self, ):
        return self.circuit.parameters()

    def set_params(self, params, ):
        self.circuit.set_params(params)

    def train(self, n_steps: int = None, cost_fn: callable = None, cost_fn_version: str = None, ):
        n_steps = n_steps or self.n_steps
        cost_fn = self.get_cost_fn(cost_fn, cost_fn_version)

        pbar = trange(n_steps, desc='Train', )
        for i in pbar:
            self.train_step(cost_fn=cost_fn)
            if i % self.eval_freq == 0 or i == n_steps - 1:
                cost = self.eval_cost(cost_fn=cost_fn)
                pbar.set_postfix({'Cost': cost})
                self.logger.log(i, cost=cost, )
            self.logger.log(i, params=self.params)
        pbar.close()
        if self.model_select == 'best':
            self.optimal_params = self.logger.optimal_params
            self.set_params(self.optimal_params)
        elif self.model_select == 'last':
            self.optimal_params = self.params
            self.set_params(self.optimal_params)
        else:
            raise ValueError(f'Unknown model selection method: {self.model_select}')

        return self.logger
