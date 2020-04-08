import tensorflow as tf
import numpy as np

class TransferFunc:
    def __init__(self, optimization_variable, interval_array):
        self.interval_array = interval_array
        self.duration = tf.reduce_sum(interval_array)
        self.optimization_variable = optimization_variable

        self.n_of_intervals = tf.shape(interval_array)[0].numpy()
        self.Legendre_basis_matrix = self.gen_Legendre_matrix(
            tf.shape(optimization_variable)[1].numpy(), self.n_of_intervals
        )

    @staticmethod
    def gen_Legendre_matrix(input_dim, output_dim):
        def Legendre_basis_vector(Legendre_coefficients):
            return np.polynomial.legendre.Legendre(
                tuple(Legendre_coefficients)
            ).linspace(output_dim)[1]

        Legendre_basis_vectors = tuple(map(
            Legendre_basis_vector, np.eye(input_dim)
        ))
        return tf.constant(
            np.row_stack(Legendre_basis_vectors), dtype=tf.float64
        )

    """
        regularize_amplitudes ensures that no individual amplitude exceeds 1
    """
    @staticmethod
    def regularize_amplitudes(amplitudes):
        amplitude_norms = tf.math.sqrt(
            tf.math.square(amplitudes[0, :]) + tf.math.square(amplitudes[1, 0])
        )
        normalization_factor = tf.math.tanh(amplitude_norms) / amplitude_norms
        return tf.math.multiply(
            normalization_factor,
            amplitudes
        )

    """
        return_physical_amplitudes transforms the input array
        self.ctrl_amplitudes into physical control amplitudes
    """

    def transformation_func(self):
        transformed_amplitudes = tf.linalg.matmul(
            self.optimization_variable, self.Legendre_basis_matrix
        )
        normalized_amplitudes = self.regularize_amplitudes(
            transformed_amplitudes
        )
        return normalized_amplitudes

    def get_amplitudes(self):
        return self.transformation_func()

    def get_intervals(self):
        return self.interval_array

    def get_duration(self):
        return self.duration

    def get_n_of_intervals(self):
        return self.n_of_intervals

    def get_max_amplitudes(self):
        return None

    """ Methods that need to be redifined  """
    def dynamic_interval_array_enabled(self):
        return False

    def get_max_intervals(self):
        return None