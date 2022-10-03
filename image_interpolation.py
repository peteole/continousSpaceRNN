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
        starts = section[:, :2]
        #print("starts", starts)
        stops = section[:, :2] + section[:, 2:]
        w_range = tf.linspace(
            starts[:, 0], stops[:, 0], self.grid_dim[0], axis=1, name="batched_x_grid")

        # shape (B,w)
        second_spacial_dim_start = tf.einsum(
            'i,j->ij', starts[:, 1], tf.ones((self.grid_dim[0],)), name="batched_y_grid_start")
        # shape (B,w,2)
        lower_stacked = tf.stack(
            [w_range, second_spacial_dim_start], axis=2, name="batched_lower_edge_stacked")
        # shape (B,w)
        second_spacial_dim_end = tf.einsum(
            'i,j->ij', stops[:, 1], tf.ones((self.grid_dim[0],)), name="batched_y_grid_end")
        # shape (B,w,2)
        upper_stacked = tf.stack(
            [w_range, second_spacial_dim_end], axis=2, name="batched_upper_edge_stacked")

        # list of tensors with shape (B,w,2) of length h
        grid = [(1.0-alpha)*lower_stacked+alpha *
                upper_stacked for alpha in [i/(self.grid_dim[1]-1) for i in range(self.grid_dim[1])]]
        interpolations = [tfp.math.batch_interp_regular_nd_grid(
            line, [0.0, 0.0], [1.0, 1.0], image, axis=1, name='batched_interpolation_line') for line in grid]
        stacked = tf.stack(interpolations, axis=2,
                           name="batched_grid_interpolation")
        return stacked

    def compute_output_shape(self, input_shape):
        return (* self.grid_dim, input_shape[-1])


class ImageSectionRNNCell(tf.keras.layers.Layer):
    """
    Args:
        grid_dim: The dimension of the grid to interpolate the image to
        units: The number of units in the LSTM cell
        Call arguments:
            image: A 4D tensor of shape (batch_size, height, width, channels)
            state_t: A tuple of the previous state of the RNN. The first element is the lstm states for all batches and the second element is the sections the network wants us to have a look at in this iteration (for all batches).
    """

    def __init__(self, grid_dim: Tuple = (16, 16), units=32, **kwargs):
        super(ImageSectionRNNCell, self).__init__(**kwargs)
        self.units = units
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
        self.state_size = (self.lstm_cell.state_size, 3)

    def call(self, image, state_t):
        #print("state_t: ", state_t)
        (last_lstm_states, section_commands) = state_t
        section_values = self.interpolator(image, section_commands)
        # if len(section_values.shape) == 3 or section_values.shape[-1] == 1:
        #     #section_values = tf.expand_dims(section_values, -1)
        #     raise ValueError("section_values has wrong shape")
        #print("section_values: ", section_values)
        processed_section_values = self.section_processor(section_values)
        # print("processed_section_values: ", processed_section_values)
        lstm_out, lstm_states = self.lstm_cell(
            processed_section_values, last_lstm_states)
        # print("lstm_out: ", lstm_out)
        next_section_command = self.next_section_command_computer(lstm_out)
        #print("next_section_command: ", next_section_command)
        return lstm_out, (lstm_states, next_section_command)

    def get_config(self):
        config = super().get_config()
        config.update({
            "grid_dim": self.grid_dim,
            "units": self.units,
        })
        return config

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        lstm_state = self.lstm_cell.get_initial_state(
            inputs, batch_size, dtype)
        section_state = tf.repeat([[0.0, 0.0, 1.0]], batch_size, axis=0)
        return [lstm_state, section_state]
