import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())


def get_model_path(env_name, actor, run_seed=None):
    return get_path(env_name, actor, "model.pkl", run_seed)


def get_trajs_path(env_name, actor, env_id, run_seed=None):
    return get_path(env_name, actor, "trajs_" + str(env_id) + ".npy", run_seed)


def get_path(env_name, actor, suffix, run_seed=None):
    dir_path = "volume/" + env_name
    fullpath = dir_path + "/" + actor + (("_" + str(run_seed)) if run_seed is not None else "") + "_" + suffix

    Path(dir_path).mkdir(parents=True, exist_ok=True)

    return fullpath
