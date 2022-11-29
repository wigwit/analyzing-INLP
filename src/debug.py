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
parser.add_argument('--load',dest='load',action='store_true', default=False,help='load the preexisting embeddings if it is already generated')
args = parser.parse_args()
#logging.basicConfig(level-logging.INFO) #turn on detailed logging

## defining GPU here
random.seed(42)

load_option = 'load' if args.load else 'save'
# layerwise = 4

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
torch.set_printoptions(edgeitems = 100, sci_mode=True)

# print('### explore whether the projection matrices are all the same ###')
# for i in range(1, 7):
#     for j in range(i, 7):
#         P_i = torch.load('../data/pmb_'+args.dataset+'/'+args.task+'_space_removed'+'_'+str(i)+'.pt')
#         P_j = torch.load('../data/pmb_'+args.dataset+'/'+args.task+'_space_removed'+'_'+str(j)+'.pt')
#         P_i = torch.tensor(P_i)
#         P_j = torch.tensor(P_j)
#         print('comparing the projection matrix of layer {} with that of layer {}. There are same values in the two projection matrices: {}'.format(i, j, torch.any(torch.eq(P_i, P_j))))
        
# ## compare layerwise performance on individual layers when different projection matrices are used
# print('### use one projection matrix on another layer ###')
# for i in range(1, 7):
#     gold_tr = DataProcessing(args.dataset, 'train')
#     gold_tr.bert_tokenize()
#     train_emb = gold_tr.get_bert_embeddings(load_option, layerwise=i)
#     emb_tr, y_tr, num_tags = gold_tr.from_sents_to_words(args.task,train_emb)

#     gold_test = DataProcessing(args.dataset,'test')
#     gold_test.bert_tokenize()

#     test_emb = gold_test.get_bert_embeddings(load_option, layerwise=i)
#     emb_test, y_test, num_tags = gold_test.from_sents_to_words(args.task,test_emb)
#     for j in range(1, 7):
#         P = torch.load('../data/pmb_'+args.dataset+'/'+args.task+'_space_removed'+'_'+str(j)+'.pt') 
#         P = torch.tensor(P)
#         P = P.type(torch.float)
        
#         print('### currently using projection matrix for layer {} on layer {} ###'.format(j, i))

#         new_emb_tr = torch.matmul(P,emb_tr.T).T 
#         new_emb_test = torch.matmul(P,emb_test.T).T

#         t_after = LinearClassifier(new_emb_tr,y_tr,num_tags)	 

#         t_after.optimize()
#         print('eval result')
#         t_after.eval(new_emb_test,y_test)


## looks like all accuracy scores are the same so now I compare the test embeddings before and after inlp
gold_test = DataProcessing(args.dataset,'test')
gold_test.bert_tokenize()

test_emb = gold_test.get_bert_embeddings(load_option, layerwise=1)
emb_test, y_test, num_tags = gold_test.from_sents_to_words(args.task,test_emb)

P_1 = torch.load('../data/pmb_'+args.dataset+'/'+args.task+'_space_removed'+'_'+str(1)+'.pt') 
P_1 = P_1.type(torch.float)
P_2 = torch.load('../data/pmb_'+args.dataset+'/'+args.task+'_space_removed'+'_'+str(2)+'.pt') 
P_2 = P_2.type(torch.float)

new_emb_test_1 = torch.matmul(P_1,emb_test.T).T

new_emb_test_2 = torch.matmul(P_2,emb_test.T).T

print('are the two post-projection embeddings the same: {}'.format(torch.any(torch.eq(new_emb_test_1, new_emb_test_2))))
print('scalar multiple?')
print(str(torch.div(new_emb_test_1, new_emb_test_2)))
print('is one of them just all zeros?')
print(str(torch.div(new_emb_test_2, new_emb_test_1)))
print('embedding after unmatching projection')
print(str(new_emb_test_2))
print('embedding after original projection')
print(str(new_emb_test_1))
 

sys.exit()
 