import tensorflow as tf
import numpy as np


class BaseDense(tf.keras.layers.Dense):
    def add_weight_kernel(self, shape, name='kernel', trainable=True):
        return self.add_weight(name=name,
                               shape=shape,
                               trainable=trainable,
                               initializer=self.kernel_initializer,
                               regularizer=self.kernel_regularizer,
                               constraint=self.kernel_constraint,
                               dtype=self.dtype)

    def add_weight_bias(self, shape, name='bias', trainable=True):
        return self.add_weight(name=name,
                               shape=shape,
                               trainable=trainable,
                               initializer=self.bias_initializer,
                               regularizer=self.bias_regularizer,
                               constraint=self.bias_constraint,
                               dtype=self.dtype)


class GraphConvs(BaseDense):
    def __init__(self, units, DADs, name=None, **kwargs):
        super(GraphConvs, self).__init__(units=units, name=name, **kwargs)
        self.DADs = tf.convert_to_tensor(DADs)
        self.num_K = len(DADs)  # K

    def build(self, input_shape):
        self.num_Fin = input_shape[-1]  # Fin
        self.num_Fout = self.units  # Fout
        self.kernel = self.add_weight_kernel(
            [self.num_K, self.num_Fin, self.num_Fout])
        self.bias = self.add_weight_bias([
            self.num_Fout,
        ]) if self.use_bias else None
        self.built = True

    def call(self, inputs):
        # DADs @ X => [?, N', Fin]@[K, N, N'] => [?, Fin, K , N]
        X = tf.tensordot(inputs, self.DADs, axes=([1], [-1]))
        # DADsX @ W => [?, Fin', K', N]@[K', Fin', Fout] => [?, N, Fout]
        X = tf.tensordot(X, self.kernel, axes=([1, 2], [1, 0]))
        if self.use_bias:
            X = tf.add(X, self.bias)
        outputs = self.activation(X)
        return outputs


class ChebyshevGraphConvs(BaseDense):
    def __init__(self, units: int, L: np.ndarray, K: int, name=None, **kwargs):
        super(ChebyshevGraphConvs, self).__init__(units=units,
                                                  name=name,
                                                  **kwargs)
        self.L = tf.convert_to_tensor(L)
        self.num_K = K  # K

    def build(self, input_shape):
        self.num_Fin = input_shape[-1]  # Fin
        self.num_Fout = self.units  # Fout
        self.kernel = self.add_weight_kernel(
            [self.num_K, self.num_Fin, self.num_Fout])
        self.bias = self.add_weight_bias([
            self.num_Fout,
        ]) if self.use_bias else None
        self.built = True

    def call(self, inputs):
        # X = Chebyshev(inputs)
        for k in range(self.num_K):
            if k == 0:
                # [?, N, Fin] => [?, 1, Fin, N]
                X0 = tf.expand_dims(tf.linalg.matrix_transpose(inputs), axis=1)
                X = X0  # B, K, Fin, N
            elif k == 1:
                # [?, 1, Fin, N']@[N, N'] => [?, 1, Fin, N]
                X1 = tf.tensordot(X0, self.L, axes=([-1], [1]))
                X = tf.concat([X, X1], axis=1)  # B, K, Fin, N
            else:
                # X2 <= 2 * L * X1 - X0
                X2 = 2 * tf.tensordot(X1, self.L, axes=([-1], [1])) - X0
                X = tf.concat([X, X2], axis=1)  # B, K, Fin, N
                X0, X1 = X1, X2
        # [?, K', Fin', N]@[K', Fin', Fout] => [?, N, Fout]
        X = tf.tensordot(X, self.kernel, axes=([1, 2], [0, 1]))
        if self.use_bias:
            X = tf.add(X, self.bias)
        outputs = self.activation(X)
        return outputs
