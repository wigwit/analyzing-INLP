import torch
from torch.utils.data import TensorDataset, DataLoader
from EvalClassifier import EvalClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(x: torch.Tensor, y: torch.Tensor, n_batches: int = 64, shuffle: bool = True) -> DataLoader:
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=n_batches, shuffle=shuffle)
    return data_loader


def train(model: torch.nn.Module, cp: str, train_loader: DataLoader, dev_loader: DataLoader, epochs: int, lr: float = 0.005):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    dev_losses = []
    for epoch in range(epochs):
        for x, y in iter(train_loader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        accuracy, dev_loss = evaluate(model, dev_loader, torch.nn.CrossEntropyLoss())
        print('Epoch: %d; training loss: %.5f; dev loss: %.5f; ' % (epoch, loss.item(), dev_loss))
        dev_losses.append(dev_loss)
        if epoch == 0:
            torch.save(model.state_dict(), cp)
        else:
            if dev_losses[epoch] < dev_losses[epoch - 1]:
                torch.save(model.state_dict(), cp)


def evaluate(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
             criterion: torch.nn.modules.loss) -> (float, float):
    correct_n = 0
    total_n = 0
    running_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            outputs = model(x)
            loss = criterion(y, outputs)
            running_loss += loss.item()
            predictions = outputs.max(1, keepdim=True)[1]
            confusion = predictions / y
            correct_n += torch.sum(confusion == 1).item()
            total_n += y.size(0)
    eval_loss = running_loss/len(test_loader)
    eval_accu = correct_n / total_n
    return eval_accu, eval_loss


def run_train(load: bool, model_path: str, x: torch.Tensor, y: torch.Tensor, dev_x: torch.Tensor, dev_y: torch.Tensor):
    model = EvalClassifier(x.shape[1], y.shape[1])
    if load:
        model.load_state_dict(torch.load(model_path, map_location=device))
    train_loader = load_data(x, y)
    dev_loader = load_data(dev_x, dev_y)
    train(model, model_path, train_loader, dev_loader, epochs=10, lr=0.0005)


def run_eval(model_path: str, eval_x: torch.Tensor, eval_y: torch.Tensor):
    model = EvalClassifier(eval_x.shape[1], eval_y.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loader = load_data(eval_x, eval_y)
    accu, loss = evaluate(model, test_loader, torch.nn.CrossEntropyLoss())
    print('model accuracy: %f; model loss: %f' % (accu, loss))


def __main__():
    load_model = False
    train_model = True
    eval_model = False
    model_path = 'model.checkpoint'
    # training_path = 'data/pmb_gold/gold_train.pkl'
    rand_x1 = torch.rand((1000, 768), device=device)
    rand_y1 = torch.randint(low=0, high=11, size=(1000, 1), dtype=torch.float, device=device)
    rand_x2 = torch.rand((100, 768), device=device)
    rand_y2 = torch.randint(low=0, high=11, size=(100, 1), dtype=torch.float, device=device)
    if train_model:
        run_train(load_model, model_path, rand_x1, rand_y1, rand_x2, rand_y2)
    #  TODO options: load model/train new model, use ccg data or use srl data, train for ccg or train for srl
    if eval_model:
        run_eval(model_path, rand_x2, rand_y2)
    #  TODO options: ccg-ccg, ccg-srl, srl-srl, srl-ccg


if __name__ == '__main__':
    __main__()


