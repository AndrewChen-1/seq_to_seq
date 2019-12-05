import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest
import numpy as np


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    """Generate a zero filled tensor with shape [batch_size, state_size]."""
    if batch_size_tensor is None or dtype is None:
        raise ValueError(
            'batch_size and dtype cannot be None while constructing initial state: '
            'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

    def create_zeros(unnested_state_size):
        flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return array_ops.zeros(init_state_size, dtype=dtype)

    if nest.is_sequence(state_size):
        return nest.map_structure(create_zeros, state_size)
    else:
        return create_zeros(state_size)


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = array_ops.shape(inputs)[0]
        dtype = inputs.dtype
    return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


class AttentionDecoderCell(keras.layers.Layer):
    def __init__(self, units,
                 hidden_state_encoder=None,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 reset_after=False,
                 **kwargs):
        super(AttentionDecoderCell, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.recurrent_activation = keras.activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = keras.initializers.get(recurrent_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = keras.constraints.get(recurrent_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.reset_after = reset_after
        self.state_size = self.units
        self.output_size = self.units
        self.x_seq = hidden_state_encoder

    def build(self, input_shape):
        """
        :param input_shape:  tuple  (batch_size, time_steps, input_dim)
        :return:
        """
        input_dim = input_shape[-1]

        self.states = [None, None]  # y, s

        # the weight matrix for attention calculation
        self.V_a = self.add_weight(shape=(self.units + self.x_seq.get_shape().as_list()[-1],),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(
            self.units + self.x_seq.get_shape().as_list()[-1], self.units + self.x_seq.get_shape().as_list()[-1]),
            name='W_a',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # the weight matrix for reset gate
        self.W_r = self.add_weight(
            shape=(self.units,),
            name='W_r',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.U_r = self.add_weight(shape=(self.unit,),
                                   name='U_r',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_r = self.add_weight(shape=(1, ),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # the weight matrix for reset gate
        self.W_z = self.add_weight(
            shape=(self.units,),
            name='W_z',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.U_z = self.add_weight(shape=(self.unit,),
                                   name='U_z',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_z = self.add_weight(shape=(1, ),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # the weight matrix for proposal
        self.W_p = self.add_weight(
            shape=(self.units,),
            name='W_p',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.U_p = self.add_weight(shape=(self.unit,),
                                   name='U_p',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_p = self.add_weight(shape=(1, ),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        # the weight matrix for output
        self.W_o = self.add_weight(
            shape=(self.units,),
            name='W_o',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.U_o = self.add_weight(shape=(self.unit,),
                                   name='U_o',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_o = self.add_weight(shape=(1, ),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        super(AttentionDecoderCell, self).build(input_shape)  # or self.built=True

    def call(self, inputs, states, training=None):
        # remember that cell deal with one time_step each time
        # yt: shape(batch_size, output_dim),  stm: shape(batch_size, units)
        stm = states[0]

        _stm = keras.backend.repeat(stm, self.x_seq.get_shape().as_list()[1])

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = keras.backend.dot(keras.backend.concatenate([_stm, self.x_seq]), self.W_a)

        # calculate the attention probabilities
        # this relates how much other time_steps contributed to this one.
        et = keras.backend.dot(keras.activations.tanh(_Wxstm),
                               keras.backend.expand_dims(self.V_a))
        at = keras.backend.exp(et)
        at_sum = keras.backend.sum(at, axis=1)
        at_sum_repeated = keras.backend.repeat(at_sum,  self.x_seq.get_shape().as_list()[1])
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = keras.backend.squeeze(keras.backend.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state

        # first calculate the "r" gate:
        rt = keras.activations.sigmoid(
            keras.backend.dot(inputs, self.W_r)
            + keras.backend.dot(stm, self.U_r)
            + keras.backend.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = keras.activations.sigmoid(
            keras.backend.dot(inputs, self.W_z)
            + keras.backend.dot(stm, self.U_z)
            + keras.backend.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = keras.activations.tanh(
            keras.backend.dot(inputs, self.W_p)
            + keras.backend.dot((rt * stm), self.U_p)
            + keras.backend.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1 - zt) * stm + zt * s_tp

        output = keras.activations.softmax(
            keras.backend.dot(inputs, self.W_o)
            + keras.backend.dot(stm, self.U_o)
            + keras.backend.dot(context, self.C_o)
            + self.b_o)

        if self.return_probabilities:
            return at, [st]
        else:
            return output, [st]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoderCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':
    a = np.arange(48).reshape(2, 2, 4, 3)  # a和b的维度有些讲究，具体查看Dot类的build方法
    b = np.arange(48).reshape(2, 6, 4)

    keras.backend.dot(keras.backend.constant(b), keras.backend.constant(a))