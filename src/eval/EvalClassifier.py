from torch import nn


class EvalClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(EvalClassifier, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size, bias=True)

    def forward(self, x):
        return self.linear(x)

