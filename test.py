from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import scipy as sp
from VLDESolver import VLDESolver
from Generators import Generators
from TransferFunc import TransferFunc
from TargetFunc import TargetFunc
import time

print(tf.__version__)

x = tf.constant(
    [[0, 1], [1, 0]], dtype=tf.complex128
)
y = tf.constant(
    [[0 + 0j, 0 - 1j], [0 + 1j, 0 + 0j]], dtype=tf.complex128
)
xL = -2 * np.pi * (0 + 1j) * np.block([
    [x.numpy(), x.numpy()],
    [np.zeros((2, 2)), x.numpy()]
])
yL = -2 * np.pi * (0 + 1j) * np.block([
    [y.numpy(), y.numpy()],
    [np.zeros((2, 2)), y.numpy()]
])
VL_matrices = tf.stack([
    tf.constant(xL, dtype=tf.complex128),
    tf.constant(yL, dtype=tf.complex128)
])

ctrl_amplitudes = tf.Variable(
    tf.zeros([2, 10], dtype=tf.float64), dtype=tf.float64
)
ctrl_amplitudes.assign(
     tf.random.uniform([2, 10], -1, 1, dtype=tf.float64)
)
interval_array = tf.ones(100, dtype=tf.float64)

generators = Generators(VL_matrices)
transfer_func = TransferFunc(ctrl_amplitudes, interval_array)
solver = VLDESolver(generators, transfer_func)
target = TargetFunc(generators, transfer_func, solver)
print(target.get_target_value())

optimizer = tf.keras.optimizers.Adam(0.01)

@tf.function
def optimization_step():
    with tf.GradientTape() as tape:
        current_target = target.get_target_value()
    gradients = tape.gradient(current_target, [ctrl_amplitudes])
    optimizer.apply_gradients(zip(gradients, [ctrl_amplitudes]))
    return current_target

steps = range(1000)

start = time.time()

for step in steps:
    current_target = optimization_step()
    print('step %2d: target=%2.5f' %
          (step, current_target))

end = time.time()

print(end-start)
