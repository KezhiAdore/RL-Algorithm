from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_size, hidden_sizes, out_size):
        super(MLP, self).__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self._layers = []
        # hidden layers
        for size in hidden_sizes:
            self._layers.append(
                nn.Sequential(
                    nn.Linear(in_features=in_size, out_features=size),
                    nn.ReLU()
                )
            )
            in_size = size
        # output layer
        self._layers.append(
            nn.Linear(in_features=in_size, out_features=out_size)
        )
        # transfer list to module list
        self._layers = nn.ModuleList(self._layers)

    def forward(self, input):
        out = input
        for layer in self._layers:
            out = layer(out)
        out=F.softmax(out)
        return out
