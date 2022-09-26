from typing import Tuple
import tensorflow as tf

import tensorflow_probability as tfp
import tensorflow_graphics as tfg
from tensorflow_graphics.geometry.representation import grid

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
