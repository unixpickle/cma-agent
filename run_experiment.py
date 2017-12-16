"""
Run CMA-ES on an RL task.
"""

from multiprocessing import Pool, cpu_count
import os

from anyrl.envs.wrappers.logs import LoggedEnv
from anyrl.models import MLP
from anyrl.rollouts import BasicRoller
from anyrl.spaces import gym_space_distribution, gym_space_vectorizer
import gym
import numpy as np
import tensorflow as tf

from cma_agent import CMATrainer

# pylint: disable=R0913
def training_loop(env_id=None,
                  layers=(8,),
                  timesteps=int(1e6),
                  batch_size=1000,
                  param_scale=0.1,
                  log_file=None):
    """
    Run CMA on the environment.
    """
    if log_file is None:
        log_file = os.path.join('results', env_id + '.monitor.csv')
    env = LoggedEnv(gym.make(env_id), log_file)
    with tf.Session() as sess:
        model = MLP(sess,
                    gym_space_distribution(env.action_space),
                    gym_space_vectorizer(env.observation_space),
                    layers)
        roller = BasicRoller(env, model, min_steps=batch_size)
        sess.run(tf.global_variables_initializer())
        trainer = CMATrainer(sess,
                             variables=[v for v in tf.trainable_variables()
                                        if 'critic' not in v.name],
                             scale=param_scale)
        steps = 0
        rewards = []
        while steps < timesteps:
            sub_steps, sub_rewards = trainer.train(roller)
            steps += sub_steps
            rewards.extend(sub_rewards)
            print('%s: steps=%d mean=%f batch_mean=%f' %
                  (env_id, steps, np.mean(rewards), np.mean(sub_rewards)))

def run_with_kwargs(kwargs):
    """
    Run an experiment with the kwargs dict.
    """
    training_loop(**kwargs)

if __name__ == '__main__':
    if not os.path.isdir('results'):
        os.mkdir('results')
    with Pool(cpu_count()) as pool:
        pool.map(run_with_kwargs,
                 [{'env_id': 'InvertedDoublePendulum-v1'},
                  {'env_id': 'InvertedPendulum-v1'},
                  {'env_id': 'HalfCheetah-v1', 'batch_size': 2000},
                  {'env_id': 'Hopper-v1'},
                  {'env_id': 'Reacher-v1'},
                  {'env_id': 'Swimmer-v1', 'batch_size': 2000},
                  {'env_id': 'Walker2d-v1', 'batch_size': 2000}])
