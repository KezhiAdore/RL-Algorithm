import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_size, hidden_sizes, out_size, final_activate=False):
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
        if final_activate:
            self._layers.append(
                nn.ReLU()
            )
        # transfer list to module list
        self._layers = nn.ModuleList(self._layers)

    def forward(self, x):
        out = x
        for layer in self._layers:
            out = layer(out)
        return out


class DuelingNet(nn.Module):

    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super(DuelingNet, self).__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.common_layers = []
        self.val_layers = []
        # hidden layers
        for size in hidden_sizes:
            self.common_layers.append(
                nn.Sequential(
                    nn.Linear(in_features=in_size, out_features=size),
                    nn.ReLU()
                )
            )
            in_size = size
        # transfer list to module list
        self.common_layers = nn.ModuleList(self.common_layers)
        self.adv_out = nn.Linear(in_size, out_size)
        self.val_out = nn.Linear(in_size, 1)

    def forward(self, x):
        for layer in self.common_layers:
            x = layer(x)
        adv = self.adv_out(x)
        val = self.val_out(x)
        adv_ave = torch.mean(adv, dim=-1, keepdim=True)
        out = val+(adv-adv_ave)
        return out


class DuelingNet2(nn.Module):

    def __init__(self, in_size, hidden_sizes, out_size) -> None:
        super(DuelingNet2, self).__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.adv_layers = []
        self.val_layers = []
        # hidden layers
        for size in hidden_sizes:
            self.adv_layers.append(
                nn.Sequential(
                    nn.Linear(in_features=in_size, out_features=size),
                    nn.ReLU()
                )
            )
            self.val_layers.append(
                nn.Sequential(
                    nn.Linear(in_features=in_size, out_features=size),
                    nn.ReLU()
                )
            )
            in_size = size
        # output layer
        self.adv_layers.append(
            nn.Linear(in_features=in_size, out_features=out_size)
        )
        self.val_layers.append(
            nn.Linear(in_features=in_size, out_features=1)
        )
        # transfer list to module list
        self.adv_layers = nn.ModuleList(self.adv_layers)
        self.val_layers = nn.ModuleList(self.val_layers)

    def forward(self, x):
        adv = x
        val = x
        for layer in self.adv_layers:
            adv = layer(adv)
        for layer in self.val_layers:
            val = layer(val)
        adv_ave = torch.mean(adv, dim=-1, keepdim=True)
        out = val+(adv-adv_ave)
        return out
