# Q-Trainer
*A High-Level API for Training Variational Quantum Circuits with Quantum Error Mitigation*


[![Unitary Fund](https://img.shields.io/badge/Supported%20By-Unitary%20Fund-FFFF00.svg)](https://unitary.fund)

**Authors:**
+ [Min Li](https://www.linkedin.com/in/min-li-310823b4) (minl2@illinois.edu)
+ [Haoxiang Wang](https://haoxiang-wang.github.io/) (hwang264@illinois.edu)

**Note:**
This software is still in active development and the API is subject to change. Please be aware that it is currently in the alpha stage and may not be fully stable.

## Example

Suppose you want to train a variational quantum circuit (with error mitigation) to solve a Max-Cut problem -- here is a simple usage example with Q-Trainer:

```python
import qtrainer
import networkx as nx 
graph = nx.random_regular_graph(3, 10) # Define a (random) graph in NetworkX
# Construct a Circuit class with Q-Trainer
circuit = qtrainer.circuits.QAOACircuit(
    graph=graph, 
    task='maxcut', # choices: 'maxcut','max_clique','max_independent_set','max_weight_cycle','max_vertex_cover'  
    depth=2, # circuit depth (number of layers) 
)
# Construct a Trainer class with Q-Trainer
trainer = qtrainer.Trainer(
    circuit=circuit, 
    device_name = 'default.mixed', # PennyLaney device name
    optimizer = 'Adam', # Optimizer name
    optimizer_config = {'stepsize': 0.1},
    shots_per_step =  100, # Number of shots per step
    n_steps = 200, # Number of training steps
    error_mitigation_method = 'zne', # Zero-noise extrapolation as default
    logger = 'tensorboard', # can be 'tensorboard' or 'wandb' or None (local CSV)
    # optional arguments: shots_per_step, error_mitigation_budget, grad_method, etc. 
)
# Launch training in one line!
logger = trainer.train()
# The circuit is now trained and can be used for Max-Cut problem solving
```

## Tutorials

+ `Tutorial-QAOA.ipynb`: Training a circuit for QAOA using Q-Trainer.
+ `Tutorial-VQE.ipynb`: Training a circuit for VQE using Q-Trainer.
+ `Tutorial-Local-Simulators.ipynb`: Using local simulators from PennyLane/Braket/Cirq/Qiskit.
+ `Tutorial-AWS-Braket.ipynb`: Using devices on AWS Braket as backend (e.g., Rigetti's quantum computers).
## Compatibility


+ **Quantum Software Platforms**
  + Interface
    + [x] PennyLane
  + Backend (Can use simulators provided by other platforms)
    + [x] Braket (Amazon)
    + [x] Cirq (Google)
    + [x] Qiskit (IBM)
+ **Hardware Compatibility**
  + [x] CPU (Simulators of PennyLane/Braket/Cirq/Qiskit)
  + [x] GPU (Simulators of PennyLane/Cirq)
  + [x] [Amazon Braket](https://aws.amazon.com/braket/)
    + [x] [Rigetti](https://www.rigetti.com/) Quantum Computer (Aspen-11/Aspen-M-2)
    + [ ] [IonQ](https://ionq.com/) Quantum Computer (Harmony)
  + [ ] [IBM Quantum](https://quantum-computing.ibm.com/)
+ **Variational Quantum Algorithms**
  + [x] [Variational Quantum Eigensolver (VQE)](https://pennylane.ai/qml/demos/tutorial_vqe.html)
  + [x] [Quantum Approximate Optimization Algorithm (QAOA)](https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html)
+ **Quantum Error Mitigation**
  + [x] [Zero-Noise Extrapolation](https://mitiq.readthedocs.io/en/stable/guide/zne.html)
  + [ ] [Probabilistic Error Cancellation](https://mitiq.readthedocs.io/en/stable/guide/pec.html)
  + [ ] [Clifford Data Regression](https://mitiq.readthedocs.io/en/stable/guide/cdr.html)
+ **Optimizers**
  + [x] Gradient Descent Family (SGD, Adam, Adagrad, etc.)
  + [x] [Quantum Natural Gradient](https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient.html)
  + [x] [Shot-Frugal Optimizer](https://pennylane.ai/qml/demos/tutorial_rosalin.html)
+ **Logging**
  + Local
    + [x] CSV, JSON, Excel
    + [x] [TensorBoard](https://www.tensorflow.org/tensorboard)
  + Online (Experiment Management Tools)
    + [x] [Weights & Biases (WandB)](https://wandb.ai/site)
  

## Installation 
Install from source repository (https://github.com/Min-Li/qtrainer)
```bash
pip install git+https://github.com/Min-Li/qtrainer.git 
```
To update the package to the latest version of this repository, please run:
```bash
pip install --upgrade --no-deps --force-reinstall git+https://github.com/Min-Li/qtrainer.git
```

## Requirements
Python 3.8 or higher

+ Required Packages (see `requirements.txt` for the full list): `numpy`, `pennylane`, `mitiq`, `tqdm`, etc.
+ Optional Packages: `pennylane-lightning[gpu]` (for GPU simulation/training), `amazon-braket-sdk` (for QPUs on Amazon Braket)

## GPU Acceleration
If you want to run quantum circuit simulate on a GPU, you can install `pennylane-lightning[gpu]` and NVIDIA [cuQuantum](https://github.com/NVIDIA/cuQuantum) SDK.
See PennyLane's [doc](https://docs.pennylane.ai/projects/lightning-gpu/en/latest/installation.html) for details about installation.

To use GPU, you just need to pass `device_name='lightning.gpu'` to the `Trainer` class. 

Notice that PennyLane does not support GPU simulation for mixed states, so you can not simulate noisy channels on GPU.  


## Noisy Simulation
To study the robustness of error-mitigated circuits, you can try various PennyLane-supported [noise models]((https://docs.pennylane.ai/en/stable/introduction/operations.html#noisy-channels)) in simulation

Currently, PennyLane only supports simulations of noisy channels using the device `default.mixed`.
You cannot use other devices such as `default.qubit` or `lightning.gpu` for noisy simulations. In addition, you can use Braket's mixed-state simulators to simulate noisy channels - see `Tutorial-Local-Simulators.ipynb` for details.

For instance, you can try PennyLane's [_noisy channels_](https://docs.pennylane.ai/en/stable/introduction/operations.html#noisy-channels) to conduct noisy simulations, such as
+ [AmplitudeDamping](https://docs.pennylane.ai/en/stable/code/api/pennylane.AmplitudeDamping.html "pennylane.AmplitudeDamping"): Single-qubit amplitude damping error channel.
+ [DepolarizingChannel](https://docs.pennylane.ai/en/stable/code/api/pennylane.DepolarizingChannel.html "pennylane.DepolarizingChannel"): 
Single-qubit symmetrically depolarizing error channel.
+ [BitFlip](https://docs.pennylane.ai/en/stable/code/api/pennylane.BitFlip.html "pennylane.BitFlip"): Single-qubit bit flip (Pauli  X) error channel.

Here we show an example of using `DeploarizingChannel` to simulate the common deploarization noise in the framework of Q-Trainer:

```python
import pennylane as qml
import qtrainer 
circuit = qtrainer.circuits.QAOACircuit(...)
noise_gate = qml.DepolarizingChannel
noise_strength = 0.1
noise_fn = qml.transforms.insert(noise_gate, noise_strength, position="all")
trainer = qtrainer.Trainer(
   circuit,
   device_name = 'default.mixed',
   noise_fn = noise_fn,
   ...
)
```
## Acknowledgements
This software project is generously supported by [Unitary Fund](https://unitary.fund/).