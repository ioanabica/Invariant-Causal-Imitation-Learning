import torch
import torch.nn as nn


# pylint: disable=redefined-builtin
class FeaturesEncoder(nn.Module):
    def __init__(self, input_size, representation_size, width):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Linear(width, representation_size),
        )

    def forward(self, x):
        return self.layers(x)


class FeaturesDecoder(nn.Module):
    def __init__(self, action_size, representation_size, width):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(representation_size + action_size, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Linear(width, representation_size),
        )

    def forward(self, x, a):
        input = torch.cat((x, a), dim=-1)
        return self.layers(input)


class ObservationsDecoder(nn.Module):
    def __init__(self, representation_size, out_size, width):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(representation_size * 2, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Linear(width, out_size),
        )

    def forward(self, x, y):
        input = torch.cat((x, y), dim=-1)
        return self.layers(input)


class EnvDiscriminator(nn.Module):
    def __init__(self, representation_size, num_envs, width):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(representation_size, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Linear(width, num_envs),
        )

    def forward(self, state):
        return self.layers(state)
