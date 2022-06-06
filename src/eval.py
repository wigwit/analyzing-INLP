import torch
from LinearClassifier import LinearClassifier
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from os.path import exists
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        predictions = model(x)
        predictions = torch.argmax(predictions, dim=1).numpy()
        y = y.numpy()
    eval_accu = accuracy_score(predictions, y)
    return eval_accu


def run_train(model_path: str, input_embeddings: torch.Tensor, output: torch.Tensor, tag_size: int):
    model = LinearClassifier(input_embeddings, output, tag_size)
    if exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.optimize()
    torch.save(model.state_dict(), model_path)


def run_eval(model_path: str, eval_x: torch.Tensor, eval_y: torch.Tensor, tag_size: int):
    model = LinearClassifier(eval_x, eval_y, tag_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    accu = evaluate(model, eval_x, eval_y)
    print('model accuracy: %f=' % accu)


def __main__():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'evaluate'), help='what to run')
    r_group = parser.add_mutually_exclusive_group(required=True)
    r_group.add_argument('-c', '--remove_ccg', action='store_true', help='remove ccg information')
    r_group.add_argument('-s', '--remove_srl', action='store_true', help='remove srl information')
    e_group = parser.add_mutually_exclusive_group(required=True)
    e_group.add_argument('-g', '--eval_ccg', action='store_true', help='evaluate on ccf tagging task')
    e_group.add_argument('-r', '--eval_srl', action='store_true', help='evaluate on srl tagging task')
    parser.add_argument('-d', '--save_dir', action='store', help='model weight location',
                        default='./src/models')

    args = parser.parse_args()

    rand_x1 = torch.rand((1000, 768), device=device)
    rand_y1 = torch.randint(low=0, high=11, size=(1000, ), device=device)
    rand_x2 = torch.rand((100, 768), device=device)
    rand_y2 = torch.randint(low=0, high=11, size=(100, ), device=device)
    num_of_class = 11

    if args.remove_ccg:
        model_path = args.save_dir + '/ccg_'
        # TODO load embedding
        # embedding = torch.load(--CCG_EMBEDDING--)
    else:
        model_path = args.save_dir + '/srl_'
        # TODO load embedding
        # embedding = torch.load(--SRL_EMBEDDING--)
    if args.eval_ccg:
        model_path = model_path + 'ccg.pt'
        # TODO: load output
        # embedding = torch.load(--CCG_OUTPUT--)
    else:
        model_path = model_path + 'srl.pt'
        # TODO: load output
        # embedding = torch.load(--SRL_OUTPUT--)

    if args.mode == 'train':
        run_train(model_path, rand_x1, rand_y1, num_of_class)
    #  TODO options: load model/train new model, use ccg data or use srl data, train for ccg or train for srl
    if args.mode == 'evaluate':
        run_eval(model_path, rand_x2, rand_y2, num_of_class)
    #  TODO options: ccg-ccg, ccg-srl, srl-srl, srl-ccg


if __name__ == '__main__':
    # rand_x1 = torch.rand((1000, 768), device=device)
    # rand_y1 = torch.randint(low=0, high=11, size=(1000, ), device=device)
    # rand_x2 = torch.rand((100, 768), device=device)
    # rand_y2 = torch.randint(low=0, high=11, size=(100, ), device=device)
    # num_of_class = 11
    # model_path = './models/m.pt'
    # run_train(model_path, rand_x1, rand_y1, num_of_class)
    __main__()

