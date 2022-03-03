import torch

from .trainable_agent import TrainableAgent


class CUDAAgent(TrainableAgent):  # pylint: disable=abstract-method
    def __init__(self, env, trajs_paths):
        super(CUDAAgent, self).__init__(
            env=env,
            trajs_paths=trajs_paths,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
