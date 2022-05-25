from torch import nn
import numpy as np


class EvalLinearClassifier(nn.Module):
    def __init__(self, weight: np.ndarray, out_dim: int, bias: np.ndarray):
        super(EvalLinearClassifier, self).__init__()
        self.linear = nn.Linear(in_features=weight.shape[0], out_features=out_dim, bias=True)
        self.linear.weight = weight
        self.linear.bias = bias

    def forward(self, x):
        return self.linear(x)

