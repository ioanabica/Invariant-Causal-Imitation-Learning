import torch.nn as nn

from .base_network import BaseNetwork


class StudentNetwork(BaseNetwork):
    def __init__(self, in_dim, out_dim, width):
        super(StudentNetwork, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width

        self.layers = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Linear(width, out_dim),
        )

    def forward(self, x):
        return self.layers(x)
