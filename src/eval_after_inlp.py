import torch
import numpy as np

from load_bert import DataProcessing
from LinearClassifier import LinearClassifier, INLPTraining
from eval_classifier import EvalClassifier

from utils import embeddingComponentBreakdown,remove_random_directions


from argparse import ArgumentParser
parser = ArgumentParser(__doc__)

parser.add_argument('--random',dest='random',action='store_true',default=False,help='nulling out random directions')
parser.add_argument('--task',dest='task',type=str,default='syn',help='choosing a task =[syn|sem] for probing task after INLP,default is syn')
parser.add_argument('--layer',dest='layer',type=int, default=-1)

args = parser.parse_args()

null_task = 'syn' if args.task == 'sem' else 'sem'

if not args.random:
    print('null task: '+null_task)
else:
    print('null task: random')

if args.layer !=-1:
    P_null = torch.load('../data/pmb_gold/'+null_task+'_space_removed_'+str(args.layer)+'.pt')
else:
    P_null = torch.load('../data/pmb_gold/'+null_task+'_space_removed.pt')
print('evaluation task: '+ args.task)



# P_sem = torch.load('../data/pmb_gold/sem_space_removed.pt')
# P_syn = torch.load('../data/pmb_gold/syn_space_removed.pt')

gold_train = DataProcessing('gold','train')
gold_train.bert_tokenize()
train_emb = gold_train.get_bert_embeddings('load',layerwise=args.layer)

emb_tr, y_tr,num_tags = gold_train.from_sents_to_words(args.task,train_emb)


gold_dev = DataProcessing('gold','dev')
gold_dev.bert_tokenize()
dev_emb = gold_dev.get_bert_embeddings('load',layerwise=args.layer)

emb_dev, y_dev, dum = gold_dev.from_sents_to_words(args.task,dev_emb)


gold_test = DataProcessing('gold','test')
gold_test.bert_tokenize()
test_emb = gold_test.get_bert_embeddings('load',layerwise=args.layer)

emb_test, y_test, dum = gold_test.from_sents_to_words(args.task,test_emb)

# print(P_sem.shape)
# print(P_syn.shape)
# new_emb_trs = embeddingComponentBreakdown(emb_tr.T.double(),P_sem,P_syn)
# new_emb_devs = embeddingComponentBreakdown(emb_dev.T.double(),P_sem,P_syn)
# sim =torch.nn.functional.cosine_similarity(new_emb_trs[1],new_emb_trs[0])
# print(sum(sim)/len(sim))

# sys.exit()
# print(new_emb_trs[4].shape)
# print(768-torch.matrix_rank(new_emb_trs[4]))

# print(new_emb_trs[2].shape)

# new_emb_tr = torch.matmul(P,emb_tr.T).T
# new_emb_dev = torch.matmul(P,emb_dev.T).T

# new_emb_tr = remove_random_directions(emb_tr.numpy(),77)
# new_emb_dev = remove_random_directions(emb_dev.numpy(),77)

# new_emb_tr = torch.tensor(new_emb_tr)
# new_emb_dev = torch.tensor(new_emb_dev)

# train_no_sem = P_sem.matmul(emb_tr.T.double()).T
# dev_no_sem = P_sem.matmul(emb_dev.T.double()).T

# train_no_syn = P_syn.matmul(emb_tr.T.double()).T
# train_no_syn = P_syn.matmul(emb_tr.T.double()).T

if not args.random:
    train_nulled = P_null.matmul(emb_tr.T.double()).T
    dev_nulled = P_null.matmul(emb_dev.T.double()).T
    test_nulled = P_null.matmul(emb_test.T.double()).T
else:
    d = 768-torch.linalg.matrix_rank(P_null).item()
    train_nulled = torch.tensor(remove_random_directions(emb_tr.numpy(),d))
    dev_nulled = torch.tensor(remove_random_directions(emb_dev.numpy(),d))
    test_nulled=torch.tensor(remove_random_directions(emb_test.numpy(),d))
t_after = LinearClassifier(train_nulled,y_tr,num_tags)
t_after.optimize(dev_x=dev_nulled,dev_y=y_dev)
_,_,test_acc=t_after.eval(test_nulled,y_test)
print(f'test accuracy for task {args.task}:{test_acc}')


# print(768-torch.matrix_rank(P_sem))
# print(768-torch.matrix_rank(P_syn))