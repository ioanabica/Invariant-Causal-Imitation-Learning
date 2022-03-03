import numpy as np
from gym import core
from numpy import linalg as LA


def get_test_mult_factors(p):
    A = np.random.rand(p, p)
    P = (A + np.transpose(A)) / 2 + p * np.eye(p)

    vals, vecs = LA.eig(P)
    w = vecs[:, 0:p]
    return w


class EnvWrapper(core.Env):  # pylint: disable=abstract-method
    def __init__(self, env, mult_factor, idx, seed):
        self._noise = 0.001
        self._mult_factor = mult_factor
        self._idx = idx
        self._env = env

        np.random.seed(seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, observations):
        noise_dims = len(self._mult_factor)
        obs_noise = np.zeros_like(observations)
        obs_noise[-noise_dims:] = np.random.randn(noise_dims) * self._noise
        spur_corr = np.matmul(observations[-noise_dims:], self._mult_factor)
        obs = np.concatenate([observations + obs_noise, spur_corr, [self._idx]])
        return obs

    def seed(self, seed):  # pylint: disable=signature-differs
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    # NOTE: Unused
    # def step(self, action):
    #     reward = 0

    #     for _ in range(self._frame_skip):
    #         time_step = self._env.step(action)
    #         reward += time_step.reward or 0
    #         done = time_step.last()
    #         if done:
    #             break
    #     obs = self._get_obs(time_step)
    #     extra["discount"] = time_step.discount
    #     return obs, reward, done, extra

    def reset(self):
        time_step = self._env.reset()
        obs = self._get_obs(time_step)
        return obs
