import tensorflow as tf

class Convert2Dto1D(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(Convert2Dto1D, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        x = inputs
        outputs = tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])
        return outputs

class GraphPool(tf.keras.layers.Layer):
    def __init__(self, num_H: int, num_W: int, name=None, **kwargs):
        super(GraphPool, self).__init__(name=name, **kwargs)
        self.H = num_H
        self.W = num_W

    def call(self, inputs):
        x = inputs
        x = tf.reshape(x, [-1, self.H, self.W, x.shape[-1]])
        x = tf.nn.pool(x,
                       window_shape=(2, 2),
                       pooling_type='MAX',
                       strides=(2, 2),
                       padding='SAME')
        outputs = tf.reshape(x, [-1, x.shape[1] * x.shape[2], x.shape[-1]])
        return outputs