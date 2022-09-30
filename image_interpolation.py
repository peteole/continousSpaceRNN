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
        print("image: ",image)
        starts = section[:, :2]
        print("starts", starts)
        stops = section[:, :2] + section[:, 2:]
        print("stops:",stops)
        queries = generate_2d_grid(
            starts=starts,
            stops=stops,
            nums=self.grid_dim,
        )
        print('queries: ', queries)
        interpolated = tfp.math.batch_interp_regular_nd_grid(
            queries, [0.0, 0.0], [1.0, 1.0], image, axis=-3)
        return interpolated

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
        print("state_t: ", state_t)
        (last_lstm_states, section_commands) = state_t
        section_values = self.interpolator(image, section_commands)
        print("section_values: ", section_values)
        processed_section_values = self.section_processor(section_values)
        print("processed_section_values: ", processed_section_values)
        lstm_out, lstm_states = self.lstm_cell(
            processed_section_values, last_lstm_states)
        print("lstm_out: ", lstm_out)
        next_section_command = self.next_section_command_computer(lstm_out)
        print("next_section_command: ", next_section_command)
        return lstm_out, (lstm_states, next_section_command)

    # def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    #     lstm_state=self.lstm_cell.get_initial_state(inputs, batch_size, dtype)
    #     print("LSTM state: ",lstm_state)
    #     return (lstm_state, tf.repeat([[0.0, 0.0, 1.0]], batch_size, axis=0))


def generate_2d_grid(starts, stops, nums:Tuple, name="grid_generate"):
    """Generates a 2D grid, similar to tf.linspace, but 2d.

    Args:
      starts: A tensor of shape `[B, 2]` containing the start coordinates of the
        grid.
      stops: A tensor of shape `[B, 2]` containing the stop coordinates of the
        grid.
      nums: A tuple of the form `[w,h]` containing the number of points in each
        dimension. Must be known at compile time.
      name: A name for this op that defaults to "grid_generate".

    Returns:
      A tensor of shape `[B, w, h, 2]` containing the 2D grid. starts and ends are inclusive. The last dimension contains the x and y coordinates of the grid points.
    """
    with tf.compat.v1.name_scope(name):
        # shape (B,2)
        starts = tf.convert_to_tensor(starts)
        # shape (B,2)
        stops = tf.convert_to_tensor(stops)
        # shape (B,w)
        w_range=tf.linspace(starts[:,0],stops[:,0],nums[0],axis=1)

        # shape (B,w)
        second_spacial_dim_start=tf.einsum('i,j->ij',starts[:,1],tf.ones((nums[0],)))
        # shape (B,w,2)
        lower_stacked=tf.stack([w_range,second_spacial_dim_start],axis=2)
        # shape (B,w)
        second_spacial_dim_end=tf.einsum('i,j->ij',stops[:,1],tf.ones((nums[0],)))
        # shape (B,w,2)
        upper_stacked=tf.stack([w_range,second_spacial_dim_end],axis=2)
        
        # liat of tensors with shape (B,w,2) of length h
        ranges=[(1.0-alpha)*lower_stacked+alpha*upper_stacked for alpha in [i/(nums[1]-1) for i in range(nums[1])]]
        # shape (B,w,h,2)
        stacked=tf.stack(ranges,axis=2)
        return stacked
                