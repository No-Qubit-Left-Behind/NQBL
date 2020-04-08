import tensorflow as tf
import numpy as np

class Generators:
    def __init__(self, generator_matrices):
        self.generator_matrices = generator_matrices

    def get_generator_matrices(self):
        return self.generator_matrices

    def get_multiplication_rule(self):
        return None