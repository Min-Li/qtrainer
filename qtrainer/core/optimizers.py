import pennylane as qml

OPTIMIZERS = {"Adam": qml.AdamOptimizer,
              "Adagrad": qml.AdagradOptimizer,
              "Momentum": qml.MomentumOptimizer,
              "NesterovMomentum": qml.NesterovMomentumOptimizer,
              "RMSProp": qml.RMSPropOptimizer,
              "Rotosolve": qml.RotosolveOptimizer,
              "Rotoselect": qml.RotoselectOptimizer,
              "QNG": qml.QNGOptimizer,
              "SGD": qml.GradientDescentOptimizer,
              "ShotAdaptive": qml.ShotAdaptiveOptimizer,
              "SPSA": qml.SPSAOptimizer,
              }

OPTIMIZER_CONFIGS = {"Adam": {"stepsize": 0.01},
                     "Adagrad": {"stepsize": 0.01},
                     "Momentum": {"stepsize": 0.01},
                     "NesterovMomentum": {"stepsize": 0.01},
                     "RMSProp": {"stepsize": 0.01},
                     "Rotosolve": {},
                     "Rotoselect": {},
                     "QNG": {},
                     "SGD": {"stepsize": 0.01},
                     "ShotAdaptive": {"min_shots": 10},
                     "SPSA": {"c0": 0.1, "c1": 0.1, "c2": 0.1, "c3": 0.1, "c4": 0.1, "niter": 100, "gamma": 0.1,
                              "eps": 0.1},
                     }


def get_optimizer(optimizer: str, **kwargs):
    if optimizer in OPTIMIZERS:
        config = OPTIMIZER_CONFIGS[optimizer]
        config.update(kwargs)
        return OPTIMIZERS[optimizer](**config)
    else:
        raise ValueError(f"Optimizer {optimizer} not supported. Please choose from {list(OPTIMIZERS.keys())}")
