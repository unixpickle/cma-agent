"""
Simple policies that are suitable for CMA-ES.
"""

from anyrl.models import Model
import numpy as np
import tensorflow as tf

class ContinuousMLP(Model):
    """
    A deterministic, continuous multi-layer perceptron.
    """
    # pylint: disable=R0913
    def __init__(self, session, action_space, obs_vectorizer,
                 layer_sizes=(8,), activation=tf.nn.relu):
        """
        Create a continuous MLP policy.

        Args:
          session: a TF session.
          action_space: the gym.Box action space.
          obs_vectorizer: the observation vectorizer.
          layer_sizes: hidden layer sizes.
          activation: the network's non-linearity.
        """
        self.session = session
        self.action_space = action_space
        self.obs_vectorizer = obs_vectorizer

        in_batch_shape = (None,) + obs_vectorizer.out_shape
        self.obs_ph = tf.placeholder(tf.float32, shape=in_batch_shape)

        flat_in_size = int(np.prod((obs_vectorizer.out_shape)))
        layer_in = tf.reshape(self.obs_ph, (-1, flat_in_size))
        for layer_idx, out_size in enumerate(layer_sizes):
            with tf.variable_scope('layer_' + str(layer_idx)):
                layer_in = tf.layers.dense(layer_in, out_size, activation=activation)

        raw_out = tf.layers.dense(layer_in, int(np.prod(action_space.shape)),
                                  activation=tf.nn.sigmoid)
        shaped_out = tf.reshape(raw_out, action_space.shape)
        self.output = (shaped_out * (action_space.high - action_space.low)) + action_space.low

    @property
    def stateful(self):
        return False

    def start_state(self, batch_size):
        return None

    def step(self, observations, states):
        obs_vecs = self.obs_vectorizer.to_vecs(observations)
        return {
            'actions': self.session.run(self.output, feed_dict={self.obs_ph: obs_vecs}),
            'states': None
        }
