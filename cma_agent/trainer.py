"""
An RL agent that optimizes a policy via CMA-ES.
"""

from math import sqrt

from anyrl.rollouts import mean_finished_reward
import cma
import numpy as np
import tensorflow as tf

# pylint: disable=R0903
class CMATrainer:
    """
    A training session.
    """
    def __init__(self, session, variables=None, scale=1.0):
        """
        Create a training session.

        Args:
          session: a TF session.
          variables: if specified, only these variables
            are optimized by the algorithm. Otherwise, all
            trainable variables are optimized.
          scale: scale for the parameter stddev.
        """
        self.session = session
        self.variables = (variables or tf.trainable_variables())
        self._placeholders = [tf.placeholder(v.dtype, shape=[int(np.prod(v.get_shape()))])
                              for v in self.variables]
        self._assigns = tf.group(*[tf.assign(v, tf.reshape(ph, tf.shape(v)))
                                   for v, ph in zip(self.variables, self._placeholders)])
        param_stddevs = []
        for var in self.variables:
            stddev = sqrt(1 / float(var.get_shape()[0].value))
            param_stddevs.extend([stddev] * int(np.prod(var.get_shape())))
        self._param_stddevs = param_stddevs
        self.cma = cma.CMAEvolutionStrategy([0] * len(param_stddevs), scale)

    def train(self, roller):
        """
        Take a step of training.

        The model parameters are updated to reflect a
        candidate solution from the solver.

        Args:
          roller: a roller that collects a batch of
            episodes to evaluate the current model.

        Returns:
          A tuple (steps, rewards):
            steps: the number of rollout steps taken.
            rewards: list of rewards from the episodes.
        """
        guesses = self.cma.ask()
        results = []
        steps = 0
        rewards = []
        for guess in guesses:
            self._put_parameters(guess)
            rollouts = roller.rollouts()
            results.append(-mean_finished_reward(rollouts))
            steps += sum([r.num_steps for r in rollouts])
            rewards.extend([r.total_reward for r in rollouts])
        self.cma.tell(guesses, results)
        return steps, rewards

    def _put_parameters(self, vector):
        """
        Put the parameters into the session.
        """
        vector = np.array(vector) * np.array(self._param_stddevs)
        feed = {}
        for var, placeholder in zip(self.variables, self._placeholders):
            size = int(np.prod(var.get_shape()))
            feed[placeholder] = vector[:size]
            vector = vector[size:]
        self.session.run(self._assigns, feed_dict=feed)
