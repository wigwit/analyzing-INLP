import torch
from eval_classifier import EvalClassifier
import pickle
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


def run_train(model_path: str, input_embeddings: torch.Tensor, output: torch.Tensor,
              dev_x: torch.Tensor, dev_y: torch.Tensor, tag_size: int):
    model = EvalClassifier(input_embeddings, output, tag_size, dev_x, dev_y)
    if exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.optimize()
    torch.save(model.state_dict(), model_path)


def run_eval(model_path: str, eval_x: torch.Tensor, eval_y: torch.Tensor,
             dev_x: torch.Tensor, dev_y: torch.Tensor, tag_size: int):
    model = EvalClassifier(eval_x, eval_y, tag_size, dev_x, dev_y)
    model.load_state_dict(torch.load(model_path, map_location=device))
    accu = evaluate(model, eval_x, eval_y)
    print('model accuracy: %f' % accu)


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
                        default='./models')

    args = parser.parse_args()

    # rand_x1 = torch.rand((1000, 768), device=device)
    # rand_y1 = torch.randint(low=0, high=11, size=(1000, ), device=device)
    # rand_x2 = torch.rand((100, 768), device=device)
    # rand_y2 = torch.randint(low=0, high=11, size=(100, ), device=device)
    # num_of_class = 11

    if args.remove_ccg:
        model_path = args.save_dir + '/ccg_'
        with open('../data/pmb_silver/rm_ccg/train.pkl', 'rb') as file:
            train_emb = pickle.load(file)[0]
        with open('../data/pmb_silver/rm_ccg/dev.pkl', 'rb') as file:
            dev_emb = pickle.load(file)[0] 
        with open('../data/pmb_silver/rm_ccg/test.pkl', 'rb') as file:
            test_emb = pickle.load(file)[0]
    else:
        model_path = args.save_dir + '/srl_'
        with open('../data/pmb_silver/rm_sem/train.pkl', 'rb') as file:
            train_emb = pickle.load(file)[0]
        with open('../data/pmb_silver/rm_sem/dev.pkl', 'rb') as file:
            dev_emb = pickle.load(file)[0] 
        with open('../data/pmb_silver/rm_sem/test.pkl', 'rb') as file:
            test_emb = pickle.load(file)[0]

    if args.eval_ccg:
        model_path = model_path + 'ccg.pt'
        with open('../data/pmb_silver/rm_ccg/train.pkl', 'rb') as file:
            train_out = pickle.load(file)[1]
        with open('../data/pmb_silver/rm_ccg/dev.pkl', 'rb') as file:
            dev_out = pickle.load(file)[1] 
        with open('../data/pmb_silver/rm_ccg/test.pkl', 'rb') as file:
            test_out = pickle.load(file)[1]
    else:
        model_path = model_path + 'srl.pt'
        with open('../data/pmb_silver/rm_ccg/train.pkl', 'rb') as file:
            train_out = pickle.load(file)[1]
        with open('../data/pmb_silver/rm_ccg/dev.pkl', 'rb') as file:
            dev_out = pickle.load(file)[1] 
        with open('../data/pmb_silver/rm_ccg/test.pkl', 'rb') as file:
            test_out = pickle.load(file)[1]
    num_of_class = torch.max(train_out).item() + 1
    if args.mode == 'train':
        print(train_emb.shape)
        print(train_out.shape)
        print(dev_emb.shape) 
        print(dev_out.shape) 
        print(num_of_class)
        run_train(model_path, train_emb, train_out, dev_emb, dev_out, num_of_class)
    if args.mode == 'evaluate':
        run_eval(model_path, test_emb, test_emb, dev_emb, dev_out, num_of_class)


if __name__ == '__main__':
    # with open('../data/pmb_silver/rm_sem/dev.pkl', 'rb') as file:
    #     dev_emb = pickle.load(file)[0] 
    #     dev_out = pickle.load(file)[1]
    # with open('../data/pmb_silver/rm_sem/test.pkl', 'rb') as file:
    #     test_emb = pickle.load(file)[0]
    #     test_out = pickle.load(file)[1]
    # num_of_class = train_out.unique(sorted=False).shape[0]

    # model_path = './models/m.pt'
    # test_emb = torch.rand((1000, 768), device=device)
    # test_out = torch.randint(low=0, high=11, size=(1000, ), device=device)
    # dev_emb = torch.rand((100, 768), device=device)
    # dev_out = torch.randint(low=0, high=11, size=(100, ), device=device)
    # num_of_class = 11
    # run_train(model_path, test_emb, test_out, dev_emb, dev_out, num_of_class)
    # run_eval(model_path, test_emb, test_out, dev_emb, dev_out, num_of_class)
    __main__()

