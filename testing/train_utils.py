import csv
import random
import warnings

import gym
import numpy as np

try:
    from paths import get_trajs_path  # noqa
except (ModuleNotFoundError, ImportError):
    from .paths import get_trajs_path  # pylint: disable=reimported

from agent import OAStableAgent
from buffer import ReplayBuffer


def save_results(results_file_path, repetition_num, match_mean, return_mean, return_std):
    with open(results_file_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow((repetition_num, match_mean, return_mean, return_std))


def save_results_mimic(results_file_path, repetition_num, match_mean, auc_score, apr_score):
    with open(results_file_path, "a") as f:
        writer = csv.writer(f)
        writer.writerow((repetition_num, match_mean, auc_score, apr_score))


def make_agent(
    env_name,
    alg_name,
    num_envs,
):
    env = gym.make(env_name)

    trajs_paths = []
    for env_id in range(num_envs):
        path = get_trajs_path(env_name, "expert", env_id)
        trajs_paths.append(path)
    algorithm = alg_name

    return OAStableAgent(
        env=env,
        trajs_paths=trajs_paths,
        algorithm=algorithm,
    )


def fill_buffer(
    trajs_path, batch_size, run_seed, traj_shift, buffer_size_in_trajs, sampling_rate, strictly_batch_data=False
):
    if strictly_batch_data:
        all_pairs = []
        for path in trajs_path:
            pairs_env = []
            trajs = np.load(path, allow_pickle=True)[()]["trajs"]

            for traj in trajs:
                for i in range(len(traj) - 2):
                    pairs_env.append((traj[i][0], traj[i][1], traj[i + 1][0], traj[i][2]))

            all_pairs += pairs_env
        random.shuffle(all_pairs)

        if len(all_pairs) < batch_size:
            warnings.warn("Buffer smaller than batch size")
            batch_size = len(all_pairs)

        state_dim = all_pairs[0][0].shape[0]

        buffer = ReplayBuffer(
            state_dim=state_dim,
            total_size=len(all_pairs),
            batch_size=batch_size,
        )

        for pair in all_pairs:
            buffer.store(
                state=pair[0], action=np.argmax(pair[1]), reward=None, next_state=pair[2], done=None, env=pair[3]
            )
    else:
        all_pairs = []
        for path in trajs_path:
            pairs_env = []
            trajs = np.load(path, allow_pickle=True)[()]["trajs"][
                run_seed * traj_shift : run_seed * traj_shift + buffer_size_in_trajs
            ]

            for traj in trajs:
                for i in range(len(traj) - sampling_rate - 1):
                    if i % sampling_rate == 0:
                        pairs_env.append((traj[i][0], traj[i][1], traj[i + sampling_rate][0], traj[i][2]))

            all_pairs += pairs_env
        random.shuffle(all_pairs)

        if len(all_pairs) - 1 < batch_size:
            warnings.warn("Buffer smaller than batch size")
            batch_size = len(all_pairs) - 1

        buffer = ReplayBuffer(
            state_dim=all_pairs[0][0].shape[0],
            total_size=len(all_pairs) - 1,
            batch_size=batch_size,
        )

        for pair in all_pairs:
            buffer.store(
                state=pair[0],
                action=pair[1],
                reward=None,
                next_state=pair[2],
                done=None,
                env=pair[3],
            )

    return buffer
