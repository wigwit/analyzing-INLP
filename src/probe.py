import torch
from transformers import AutoTokenizer, AutoModel
import sys
import logging
import pandas as pd
import itertools
import pdb
import json
import pickle
from typing import Dict, List
from utils import bert_tokenization, get_projection_to_intersection_of_nullspaces, get_rowspace_projection, train_linear_classifier
import numpy as np
from tqdm import tqdm
import random

from load_bert import DataProcessing
from LinearClassifier import LinearClassifier, INLPTraining
from eval_classifier import EvalClassifier

from argparse import ArgumentParser
parser = ArgumentParser(__doc__)

parser.add_argument('--dataset',dest='dataset',type=str, default='gold',help='choosing either gold or silver standard data')
parser.add_argument('--task',dest='task',type=str,default='syn',help='choosing a task =[syn|sem] for INLP,default is syn')
parser.add_argument('--layer',dest='layer',type=int,default=-1,help='choosing if running on layer embeddings')
parser.add_argument('--load',dest='load',action='store_true', default=False,help='load the preexisting embeddings if it is already generated')
args = parser.parse_args()



random.seed(42)

load_option = 'load' if args.load else 'save'
layerwise= args.layer


# defining GPU here
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')


print('Loading Dataset: '+args.dataset)

train_data = DataProcessing(args.dataset,'train')
train_data.bert_tokenize()
train_emb_by_sents = train_data.get_bert_embeddings(load_option, layerwise=layerwise)

## all in cpu right now
train_emb,train_y , num_tags = train_data.from_sents_to_words(args.task,train_emb_by_sents)

dev_data = DataProcessing(args.dataset,'dev')
dev_data.bert_tokenize()
dev_emb_by_sents = dev_data.get_bert_embeddings(load_option, layerwise=layerwise)

## all in cpu right now
dev_emb, dev_y, num_tags = dev_data.from_sents_to_words(args.task,dev_emb_by_sents)

min_acc = np.bincount(train_y.numpy()).max()/train_y.shape[0]
print(min_acc)

print('Running INLP Loop for task: '+args.task)
## calling INLP
inlp_syn = INLPTraining(train_emb,train_y,num_tags)
inlp_syn = inlp_syn.to(device)
P,P_is,Ws,Ps=inlp_syn.run_INLP_loop(20,dev_x=dev_emb,dev_y=dev_y,min_acc=min_acc)
print(f'the rank of P is :{np.linalg.matrix_rank(P)}')
print(f'the rank gets removed :{768-np.linalg.matrix_rank(P)}')


## preparing test data
test_data = DataProcessing(args.dataset,'test')
test_data.bert_tokenize()
test_emb_by_sents = test_data.get_bert_embeddings(load_option, layerwise=layerwise)
test_emb, test_y, num_tags = test_data.from_sents_to_words(args.task,test_emb_by_sents)

_,_,test_acc=inlp_syn.eval(test_emb,test_y)

print(f'the final test accuracy is: {test_acc}')
print('############# AFTER INLP #############')

for index,m in enumerate(Ps):
	print(f'the rank of P_{index} is :{np.linalg.matrix_rank(m)}')
	#print(f'the rank of weight matrix is :{np.linalg.matrix_rank(Ws[index])}')
	m = torch.tensor(m)
	projection = torch.matmul(m,dev_emb.T.double()).T
	# print(projection.shape)
	#print(f'{torch.linalg.matrix_rank(projection)}')
	print(f'norm of projection is :{torch.linalg.matrix_norm(m)}')

# 	if index == len(Ps)-1:
# 		m1 = projection
# 	if index == len(Ps)-2:
# 		m2 = projection

# m_f = torch.cat((m1,m2))
# print(torch.linalg.matrix_rank(m_f))



P_t = torch.tensor(P)
# #print(torch.matrix_rank(P_t))
if layerwise == -1:
	save_path_inlp = '../data/pmb_'+args.dataset+'/'+args.task+'_space_removed.pt'
else:
	save_path_inlp = '../data/pmb_'+args.dataset+'/'+args.task+'_space_removed'+'_'+str(layerwise)+'.pt'
torch.save(P_t,save_path_inlp)

