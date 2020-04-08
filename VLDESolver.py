import tensorflow as tf
import numpy as np
from Generators import Generators
from TransferFunc import TransferFunc

class VLDESolver:
    def __init__(self, generators, transfer_func, target_func=None,
                 solver_type='expm_propagate', solution_indices=None):

        """ generators is an instance of VLMatrix class  """
        self.generators = generators
        self.generator_matrices = generators.get_generator_matrices()
        self.custom_multiplication_rule = None


        """ transfer_func is an instance of TransferFunc class  """
        self.transfer_func = transfer_func
        self.n_of_intervals = transfer_func.get_n_of_intervals()
        self.max_amplitudes = transfer_func.get_max_amplitudes()

        if transfer_func.dynamic_interval_array_enabled():
            self.max_intervals = transfer_func.get_max_intervals()
        else:
            self.max_intervals = transfer_func.get_intervals()


        self.solution_indices = solution_indices
        """ target_func is an instance of TargetFunc class  """
        self.target_func = target_func
        if target_func is not None:
            self.solution_indices = target_func.get_solution_indices()

        """ TODO: add sparsness and type variables (sparseness should be asked from generators object) """

        self.solver_type = solver_type
        """
            initiate all variables necessary for 'expm_propagate' solver
        """
        if solver_type == 'expm_propagate':
            if self.solution_indices is None:
                self.multi_dot_array = self.gen_multi_dot_array(
                    self.n_of_intervals
                )

    """
        self.gen_multi_dot_array() determines the necessity for the extra
        matrix multiplication step in the recursive step of
        self.expm_propagate() when the intermediate computation array has
        length not divisible by 2
    """
    @staticmethod
    def gen_multi_dot_array(n_of_intervals):
        if n_of_intervals > 1:
            return ([bool(np.mod(n_of_intervals, 2))] +
                    VLDESolver.gen_multi_dot_array(
                        np.floor(n_of_intervals / 2)
                    ))
        return []

    @staticmethod
    def matrix_exp_pade3(matrix, multiplication_rule=None):
        """3rd-order Pade approximant for matrix exponential."""
        b = [120.0, 60.0, 12.0]
        b = [tf.constant(x, matrix.dtype) for x in b]
        ident = tf.linalg.eye(
            tf.shape(matrix)[-2],
            batch_shape=tf.shape(matrix)[:-2],
            dtype=matrix.dtype)
        matrix_2 = tf.linalg.matmul(matrix, matrix)
        tmp = matrix_2 + b[1] * ident
        matrix_u = tf.linalg.matmul(matrix, tmp)
        matrix_v = b[2] * matrix_2 + b[0] * ident
        return matrix_u, matrix_v

    @staticmethod
    def matrix_exp_pade5(matrix, multiplication_rule=None):
        """5th-order Pade approximant for matrix exponential."""
        b = [30240.0, 15120.0, 3360.0, 420.0, 30.0]
        b = [tf.constant(x, matrix.dtype) for x in b]
        ident = tf.linalg.eye(
            tf.shape(matrix)[-2],
            batch_shape=tf.shape(matrix)[:-2],
            dtype=matrix.dtype)
        matrix_2 = tf.linalg.matmul(matrix, matrix)
        matrix_4 = tf.linalg.matmul(matrix_2, matrix_2)
        tmp = matrix_4 + b[3] * matrix_2 + b[1] * ident
        matrix_u = tf.linalg.matmul(matrix, tmp)
        matrix_v = b[4] * matrix_4 + b[2] * matrix_2 + b[0] * ident
        return matrix_u, matrix_v

    @staticmethod
    def matrix_exp_pade7(matrix, multiplication_rule=None):
        """7th-order Pade approximant for matrix exponential."""
        b = [17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0]
        b = [tf.constant(x, matrix.dtype) for x in b]
        ident = tf.linalg.eye(
            tf.shape(matrix)[-2],
            batch_shape=tf.shape(matrix)[:-2],
            dtype=matrix.dtype)
        matrix_2 = tf.linalg.matmul(matrix, matrix)
        matrix_4 = tf.linalg.matmul(matrix_2, matrix_2)
        matrix_6 = tf.linalg.matmul(matrix_4, matrix_2)
        tmp = matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 + b[1] * ident
        matrix_u = tf.linalg.matmul(matrix, tmp)
        matrix_v = b[6] * matrix_6 + b[4] * matrix_4 + b[2] * matrix_2 + b[0] * ident
        return matrix_u, matrix_v

    @staticmethod
    def matrix_exp_pade9(matrix, multiplication_rule=None):
        """9th-order Pade approximant for matrix exponential."""
        b = [
            17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0,
            2162160.0, 110880.0, 3960.0, 90.0
        ]
        b = [tf.constant(x, matrix.dtype) for x in b]
        ident = tf.linalg.eye(
            tf.shape(matrix)[-2],
            batch_shape=tf.shape(matrix)[:-2],
            dtype=matrix.dtype)
        matrix_2 = tf.linalg.matmul(matrix, matrix)
        matrix_4 = tf.linalg.matmul(matrix_2, matrix_2)
        matrix_6 = tf.linalg.matmul(matrix_4, matrix_2)
        matrix_8 = tf.linalg.matmul(matrix_6, matrix_2)
        tmp = (
                matrix_8 + b[7] * matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 +
                b[1] * ident)
        matrix_u = tf.linalg.matmul(matrix, tmp)
        matrix_v = (
                b[8] * matrix_8 + b[6] * matrix_6 + b[4] * matrix_4 + b[2] * matrix_2 +
                b[0] * ident)
        return matrix_u, matrix_v

    @staticmethod
    def matrix_exp_pade13(matrix, multiplication_rule=None):
        """13th-order Pade approximant for matrix exponential."""
        b = [
            64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
            1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0,
            33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0
        ]
        b = [tf.constant(x, matrix.dtype) for x in b]
        ident = tf.linalg.eye(
            tf.shape(matrix)[-2],
            batch_shape=tf.shape(matrix)[:-2],
            dtype=matrix.dtype)
        matrix_2 = tf.linalg.matmul(matrix, matrix)
        matrix_4 = tf.linalg.matmul(matrix_2, matrix_2)
        matrix_6 = tf.linalg.matmul(matrix_4, matrix_2)
        tmp_u = (
                tf.linalg.matmul(matrix_6, matrix_6 + b[11] * matrix_4 + b[9] * matrix_2) +
                b[7] * matrix_6 + b[5] * matrix_4 + b[3] * matrix_2 + b[1] * ident)
        matrix_u = tf.linalg.matmul(matrix, tmp_u)
        tmp_v = b[12] * matrix_6 + b[10] * matrix_4 + b[8] * matrix_2
        matrix_v = (
                tf.linalg.matmul(matrix_6, tmp_v) + b[6] * matrix_6 + b[4] * matrix_4 +
                b[2] * matrix_2 + b[0] * ident)
        return matrix_u, matrix_v

    @staticmethod
    def matrix_exponential(input, multiplication_rule=None):
        matrix = input
        l1_norm = tf.math.reduce_max(
            tf.math.reduce_sum(
                tf.math.abs(matrix),
                axis=tf.size(tf.shape(matrix)) - 2),
            axis=-1)
        const = lambda x: tf.constant(x, l1_norm.dtype)

        def nest_where(vals, cases):
            assert len(vals) == len(cases) - 1
            if len(vals) == 1:
                return tf.where(
                    tf.math.less(l1_norm, const(vals[0]))[:, tf.newaxis, tf.newaxis],
                    cases[0], cases[1]
                )
            else:
                return tf.where(
                    tf.math.less(l1_norm, const(vals[0]))[:, tf.newaxis, tf.newaxis],
                    cases[0], nest_where(vals[1:], cases[1:])
                )

        maxnorm = const(5.371920351148152)
        squarings = tf.math.maximum(
            tf.math.floor(
                tf.math.log(l1_norm / maxnorm) / tf.math.log(const(2.0))), 0)
        u3, v3 = VLDESolver.matrix_exp_pade3(matrix, multiplication_rule)
        u5, v5 = VLDESolver.matrix_exp_pade5(matrix, multiplication_rule)
        u7, v7 = VLDESolver.matrix_exp_pade7(matrix, multiplication_rule)
        u9, v9 = VLDESolver.matrix_exp_pade9(matrix, multiplication_rule)
        u13, v13 = VLDESolver.matrix_exp_pade13(matrix / tf.math.pow(
            tf.constant(2.0, dtype=matrix.dtype),
            tf.cast(
                squarings,
                matrix.dtype))[:, tf.newaxis, tf.newaxis], multiplication_rule)
        conds = (1.495585217958292e-002, 2.539398330063230e-001,
                 9.504178996162932e-001, 2.097847961257068e+000)
        u = nest_where(conds, (u3, u5, u7, u9, u13))
        v = nest_where(conds, (v3, v5, v7, v9, v13))

        numer = u + v
        denom = -u + v
        result = tf.linalg.solve(denom, numer)
        max_squarings = tf.math.reduce_max(squarings)

        i = const(0.0)
        c = lambda i, r: tf.math.less(i, max_squarings)

        def b(i, r):
            return i + 1, tf.where(
                tf.math.less(i, squarings)[:, tf.newaxis, tf.newaxis],
                tf.linalg.matmul(r, r), r
            )

        _, result = tf.while_loop(c, b, [i, result])
        return result

    """
        expm_propagate computes the final propagator by recursively multiplying
        each odd element in the list of matrices with each even element --
        if the length of the array is not divisible by 2 an extra computation
        step is added
    """
    def expm_propagate(self, interval_array, amplitude_array):

        exponents = tf.linalg.tensordot(
            tf.cast(
                tf.math.multiply(interval_array, amplitude_array),
                dtype=tf.complex128
            ),
            self.generator_matrices,
            [[0], [0]]
        )

        # products = self.matrix_exponential(exponents)
        products = tf.linalg.expm(exponents) # should be replaced
        for is_odd in self.multi_dot_array:
            if is_odd:
                last_element = products[-1:, :, :]
                products = tf.linalg.matmul(
                    products[1::2, :, :], products[0:-1:2, :, :]
                )
                products = tf.concat([
                    products[0:-1, :, :],
                    tf.linalg.matmul(last_element, products[-1:, :, :])
                ], 0)
            else:
                products = tf.linalg.matmul(
                    products[1::2, :, :], products[0::2, :, :]
                )
        return tf.squeeze(products)


    @tf.function
    def get_VLDE_solution(self):
        if self.solver_type == 'expm_propagate':
            return self.expm_propagate(
                self.transfer_func.get_intervals(),
                self.transfer_func.get_amplitudes()
            )