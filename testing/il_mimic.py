import argparse
import os
import pickle

import gym
import numpy as np
import pandas as pd

try:
    from paths import get_model_path  # noqa
except (ModuleNotFoundError, ImportError):
    from .paths import get_model_path  # pylint: disable=reimported

from contrib.energy_model import EnergyModel
from network import (
    EnvDiscriminator,
    FeaturesDecoder,
    FeaturesEncoder,
    MineNetwork,
    ObservationsDecoder,
    StudentNetwork,
)
from student import BaseStudent, ICILStudent
from testing.train_utils import fill_buffer, make_agent, save_results_mimic


# pylint: disable=redefined-outer-name
def make_student(run_seed: int, config) -> BaseStudent:
    trajs_path = config["TRAIN_TRAJ_PATH"]
    model_path = get_model_path(config["ENV"], "student_" + config["ALG"], run_seed=run_seed)

    state_dim = config["STATE_DIM"]
    action_dim = config["ACTION_DIM"]
    num_training_envs = config["NUM_TRAINING_ENVS"]

    # run_seed = run_seed
    batch_size = config["BATCH_SIZE"]
    buffer_size_in_trajs = config["NUM_TRAJS_GIVEN"]

    adam_alpha = config["ADAM_ALPHA"]

    env = gym.make("CartPole-v1")  # This is needed such the student code doesn't break.
    teacher = make_agent("CartPole-v1", "dqn", config["NUM_TRAINING_ENVS"])
    teacher.load_pretrained()

    buffer = fill_buffer(
        trajs_path=config["TRAIN_TRAJ_PATH"],
        batch_size=batch_size,
        run_seed=run_seed,
        traj_shift=None,
        buffer_size_in_trajs=buffer_size_in_trajs,
        sampling_rate=None,
        strictly_batch_data=True,
    )

    energy_model = EnergyModel(
        in_dim=state_dim,
        width=config["MLP_WIDTHS"],
        batch_size=batch_size,
        adam_alpha=adam_alpha,
        buffer=buffer,
        sgld_buffer_size=config["SGLD_BUFFER_SIZE"],
        sgld_learn_rate=config["SGLD_LEARN_RATE"],
        sgld_noise_coef=config["SGLD_NOISE_COEF"],
        sgld_num_steps=config["SGLD_NUM_STEPS"],
        sgld_reinit_freq=config["SGLD_REINIT_FREQ"],
    )
    energy_model.train(num_updates=config["NUM_STEPS_TRAIN_ENERGY_MODEL"])

    causal_features_encoder = FeaturesEncoder(
        input_size=state_dim, representation_size=config["REP_SIZE"], width=config["MLP_WIDTHS"]
    )

    causal_features_decoder = FeaturesDecoder(
        action_size=action_dim, representation_size=config["REP_SIZE"], width=config["MLP_WIDTHS"]
    )

    observations_decoder = ObservationsDecoder(
        representation_size=config["REP_SIZE"], out_size=state_dim, width=config["MLP_WIDTHS"]
    )

    policy_network = StudentNetwork(in_dim=config["REP_SIZE"], out_dim=action_dim, width=config["MLP_WIDTHS"])

    env_discriminator = EnvDiscriminator(
        representation_size=config["REP_SIZE"], num_envs=config["NUM_TRAINING_ENVS"], width=config["MLP_WIDTHS"]
    )

    noise_features_encoders = [
        FeaturesEncoder(input_size=state_dim, representation_size=config["REP_SIZE"], width=config["MLP_WIDTHS"])
        for i in range(num_training_envs)
    ]
    noise_features_decoders = [
        FeaturesDecoder(action_size=action_dim, representation_size=config["REP_SIZE"], width=config["MLP_WIDTHS"])
        for i in range(num_training_envs)
    ]

    mine_network = MineNetwork(x_dim=config["REP_SIZE"], z_dim=config["REP_SIZE"], width=config["MLP_WIDTHS"])

    return ICILStudent(
        env=env,
        trajs_paths=trajs_path,
        model_path=model_path,
        num_training_envs=num_training_envs,
        teacher=teacher,
        causal_features_encoder=causal_features_encoder,
        noise_features_encoders=noise_features_encoders,
        causal_features_decoder=causal_features_decoder,
        noise_features_decoders=noise_features_decoders,
        observations_decoder=observations_decoder,
        env_discriminator=env_discriminator,
        policy_network=policy_network,
        energy_model=energy_model,
        mine_network=mine_network,
        buffer=buffer,
        adam_alpha=adam_alpha,
    )


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", default=0, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg()

    config = {
        "ENV": "MIMIC",
        "ALG": "ICILStudent",
        "NUM_TRAINING_ENVS": 2,
        "REP_SIZE": 32,
        "NUM_STEPS_TRAIN": 3000,
        "NUM_REPETITIONS": 10,
        "BATCH_SIZE": 128,
        "MLP_WIDTHS": 64,
        "ADAM_ALPHA": 0.0005,
        "SGLD_BUFFER_SIZE": 10000,
        "SGLD_LEARN_RATE": 0.01,
        "SGLD_NOISE_COEF": 0.01,
        "SGLD_NUM_STEPS": 50,
        "SGLD_REINIT_FREQ": 0.05,
        "NUM_STEPS_TRAIN_ENERGY_MODEL": 1000,
        "TRAIN_TRAJ_PATH": ["volume/MIMIC/train_mimic_ventilator_0.npy", "volume/MIMIC/train_mimic_ventilator_1.npy"],
        "TEST_TRAJ_PATH": "volume/MIMIC/test_mimic_ventilator.npy",
        "NUM_TRAJS_GIVEN": 4000,
        "STATE_DIM": 228,
        "ACTION_DIM": 2,
    }

    print("Config: %s" % config)

    TRIAL = args.trial
    print("Trial number %s" % TRIAL)

    results_dir_base = "testing/results/"
    results_dir = os.path.join(results_dir_base, config["ENV"], config["ALG"])

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    config_file = "trial_" + str(TRIAL) + "_" + "config.pkl"

    results_file_name = "trial_" + str(TRIAL) + "_" + "results.csv"
    results_file_path = os.path.join(results_dir, results_file_name)

    if os.path.exists(os.path.join(results_dir, config_file)):
        raise NameError("CONFIG file already exists %s. Choose a different trial number." % config_file)
    pickle.dump(config, open(os.path.join(results_dir, config_file), "wb"))

    for run_seed in range(config["NUM_REPETITIONS"]):
        print("Run %s out of %s" % (run_seed + 1, config["NUM_REPETITIONS"]))

        student = make_student(run_seed, config)
        student.train(num_updates=config["NUM_STEPS_TRAIN"])
        action_match, auc_score, apr_score = student.test_batch_data(config["TEST_TRAJ_PATH"])
        result = (action_match, auc_score, apr_score)

        print(
            "Results on test environment for run %s: accuracy %.3f, auc %.3f, apr %.3f"
            % (run_seed + 1, action_match, auc_score, apr_score)
        )

        save_results_mimic(results_file_path, run_seed, action_match, auc_score, apr_score)

    results_mimic = pd.read_csv(
        "testing/results/" + config["ENV"] + "/" + config["ALG"] + "/trial_" + str(TRIAL) + "_results.csv", header=None
    )

    print(
        "Average results for "
        + str(config["NUM_REPETITIONS"])
        + " repetitions: accuracy %.3f, auc %.3f, apr %.3f"
        % (np.mean(results_mimic[1].values), np.mean(results_mimic[2].values), np.mean(results_mimic[3].values))
    )
