from typing import Tuple
import tensorflow as tf

import tensorflow_probability as tfp
import tensorflow_graphics as tfg
from tensorflow_graphics.geometry.representation import grid
layers = tf.keras.layers

i = tf.keras.Input(shape=(28, 28, 1))


class ImageInterpolator(tf.keras.layers.Layer):

    def __init__(self, grid_dim: Tuple = (16, 16), **kwargs):
        super(ImageInterpolator, self).__init__(**kwargs)
        self.grid_dim = grid_dim
    """
    Args:
        image: A 4D tensor of shape (batch_size, height, width, channels)
        section: A 2D tensor of shape (batch_size, 3) containing the coordinates and scale of the section. (Sensible) coordinates are in the range [0, 1] and scale is in the range [0, 1] (0 means super small, 1 means normal size). Outside coordinates are extended constantly.
    Returns: A 4D tensor of shape (batch_size, grid_dim[0], grid_dim[1], channels)
    """

    def call(self, image, section):
        queries = grid.generate(
            starts=section[:, :2],
            stops=section[:, :2] + section[:, 2:],
            nums=self.grid_dim,
        )
        interpolated = tfp.math.batch_interp_regular_nd_grid(
            queries, [0.0, 0.0], [1.0, 1.0], image, axis=-3)
        return interpolated

    def compute_output_shape(self, input_shape):
        return (* self.grid_dim, input_shape[-1])

"""
Call arguments:
    image: A 4D tensor of shape (batch_size, height, width, channels)
    state_t: A tuple of the previous state of the RNN. The first element is the lstm states for all batches and the second element is the sections the network wants us to have a look at in this iteration (for all batches).
"""
class ImageSectionRNNCell(tf.keras.layers.Layer):
    """
    Args:
        grid_dim: The dimension of the grid to interpolate the image to
        units: The number of units in the LSTM cell
    """
    def __init__(self, grid_dim: Tuple = (16, 16), units=32, **kwargs):
        super(ImageSectionRNNCell, self).__init__(**kwargs)
        self.grid_dim = grid_dim
        self.interpolator = ImageInterpolator(grid_dim=self.grid_dim)
        self.section_processor = tf.keras.Sequential([
            layers.Conv2D(3, 3, activation='relu'),
            layers.Conv2D(3, 3, activation='relu'),
            layers.Flatten(),
            layers.Dense(units, activation='relu')
        ])
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.next_section_command_computer = tf.keras.Sequential([
            layers.Dense(units, activation='relu'),
            layers.Dense(3, activation='sigmoid')
        ])

    def call(self, image, state_t):
        (last_lstm_states, section_commands) = state_t
        section_values = self.interpolator(image, section_commands)
        processed_section_values = self.section_processor(section_values)
        lstm_out, lstm_states = self.lstm_cell(
            processed_section_values, last_lstm_states)
        next_section_command = self.next_section_command_computer(lstm_out)
        return lstm_out, (lstm_states, next_section_command)

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return (self.lstm_cell.get_initial_state(inputs, batch_size, dtype), tf.repeat([0.0, 0.0, 1.0], batch_size, axis=0))
