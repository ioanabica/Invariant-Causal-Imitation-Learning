import torch.nn as nn


class BaseNetwork(nn.Module):
    def __init__(self):  # pylint: disable=useless-super-delegation
        super(BaseNetwork, self).__init__()

    def forward(self, x):
        raise NotImplementedError
