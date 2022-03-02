# import gym
import numpy as np
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch import nn, optim
# import torch.nn.functional as F
# import warnings

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
# from tqdm import tqdm

from agent import SerializableAgent  # BaseAgent
# from buffer import ReplayBuffer
# from contrib.env_wrapper import EnvWrapper


# pylint: disable=arguments-differ
class BaseStudent(SerializableAgent):
    def __init__(self, env, trajs_paths, model_path, teacher, buffer):
        super(BaseStudent, self).__init__(
            env=env,
            trajs_paths=trajs_paths,
            model_path=model_path,
        )
        self.teacher = teacher
        self.buffer = buffer
        self.trajs_paths = trajs_paths

    def matchup(self):
        samples = self.buffer.sample_all()
        state = samples['state']
        action = samples['action']

        action_hat = np.array([self.select_action([s]) for s in state])
        match_samp = np.equal(action, action_hat)

        return match_samp

    def rollout_env(self, env_wrapper):
        state = self.env.reset()
        state_env = env_wrapper._get_obs(state)

        traj = []
        match = []
        retvrn = 0

        done = False

        while not done:
            action = self.select_action(state_env)
            reward, next_state, done = self.perform_action(action)

            next_state_env = env_wrapper._get_obs(next_state)

            traj += [(state_env, action, env_wrapper._idx)]
            match += [action == self.teacher.select_action(state)]
            retvrn += reward

            state = next_state
            state_env = next_state_env

        return traj, match, retvrn

    def test(self, num_episodes, env_wrapper):
        self.test_mode = True

        trajs = []
        matches = []
        returns = []

        for episode_index in range(num_episodes):
            traj, match, retvrn = self.rollout_env(env_wrapper)

            trajs += [traj]
            matches += match
            returns += [retvrn]

        return np.sum(matches) / len(matches), np.mean(returns), np.std(returns)

    def test_batch_data(self, test_trajs_path):
        trajs = np.load(test_trajs_path, allow_pickle=True)[()]['trajs']

        true_actions = []
        predicted_actions = []
        predicted_actions_prob = []

        for traj in trajs:
            for i, pair in enumerate(traj):
                state = pair[0]
                true_act = pair[1]
                try:
                    pred_act, pred_act_prob = self.select_action(state, eval_mode=True)  # pylint: disable=unexpected-keyword-arg
                except TypeError as e:
                    raise RuntimeError("Expected `self` to be `ICILStudent`.") from e

                true_actions.append(true_act)
                predicted_actions.append(pred_act)
                predicted_actions_prob.append(pred_act_prob)

        accuracy = accuracy_score(y_true=true_actions, y_pred=predicted_actions)
        auc_score = roc_auc_score(y_true=true_actions, y_score=predicted_actions_prob, multi_class='ovr')
        apr_score = average_precision_score(y_true=true_actions, y_score=predicted_actions_prob)

        return accuracy, auc_score, apr_score

    def serialize(self):
        raise NotImplementedError

    def deserialize(self):
        raise NotImplementedError
