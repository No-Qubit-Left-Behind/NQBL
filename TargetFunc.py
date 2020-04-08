import tensorflow as tf
import numpy as np
from Generators import Generators
from TransferFunc import TransferFunc
from VLDESolver import VLDESolver

class TargetFunc:
    def __init__(self, generators, transfer_func, solver):
        self.generators = generators
        self.transfer_func = transfer_func
        self.solver = solver
        self.duration = transfer_func.get_duration()

    def get_target_value(self):
        x = tf.constant(
            [[0, 1], [1, 0]], dtype=tf.complex128
        )
        propagator = self.solver.get_VLDE_solution()
        tr = tf.linalg.trace(tf.linalg.matmul(x, propagator[0:2, 0:2]))
        """
            infidelity part in the target
        """
        infidelity = 1 - tf.math.real(tr * tf.math.conj(tr)) / (2 ** 2)
        """
            robustness term in the target
        """
        norm_squared = 1 / ((2 * np.pi * self.duration) ** 2) / 2 * (
            tf.math.real(
                tf.linalg.trace(
                    tf.linalg.matmul(
                        propagator[0:2, 2:4],
                        propagator[0:2, 2:4],
                        adjoint_b=True
                    )
                )
            )
        )
        return 0.5 * infidelity + 0.5 * norm_squared
