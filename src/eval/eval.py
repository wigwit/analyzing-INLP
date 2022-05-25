import torch
import torch.utils.data
import pickle
import EvalLinearClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('data/test.pkl') as file:
    test_df = pickle.load(file)

# TODO load data, get weight
m = EvalLinearClassifier(weight=, out_dim=, bias=)


def evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader) -> float:
    correct_n = 0
    total_n = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            predictions = outputs.max(1, keepdim=True)[1]
            confusion = predictions / y
            correct_n += torch.sum(confusion == 1).item()
            total_n += y.size(0)
    return correct_n / total_n
