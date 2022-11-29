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

investigate_layer = 3
projection_layer = 5


device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')


# print('Loading Dataset: '+args.dataset)

# gold_train = DataProcessing(args.dataset,'train')
# gold_train.bert_tokenize()
# train_emb = gold_train.get_bert_embeddings(load_option, layerwise=layerwise)

# ## all in cpu right now
# emb_tr, y_tr_sem, num_tags_sem = gold_train.from_sents_to_words(args.task,train_emb)
# dum, y_tr_syn,num_tags_syn = gold_train.from_sents_to_words(args.task,train_emb)

# gold_dev = DataProcessing(args.dataset,'test')
# gold_dev.bert_tokenize()
# dev_emb = gold_dev.get_bert_embeddings(load_option, layerwise=layerwise)

# ## all in cpu right now
# emb_dev, y_dev_sem, num_tags_sem = gold_dev.from_sents_to_words(args.task,dev_emb)
# dum,y_dev_syn,num_tags_syn =gold_dev.from_sents_to_words(args.task,dev_emb)


# if torch.cuda.is_available():
# 	torch.cuda.empty_cache()

# print(num_tags_syn)
# print(np.unique(y_dev_syn.numpy()))

# print('Testing Linear Classifer for task'+args.task)

# t = LinearClassifier(emb_tr,y_tr_sem,num_tags_sem)
# t.optimize()

# t.eval(emb_dev,y_dev_sem)

# min_acc = np.bincount(y_tr_sem.numpy()).max()/y_tr_sem.shape[0]
# print(min_acc)

# print('Testing INLP Loop for task: '+args.task)
# ## calling INLP
# inlp_syn = INLPTraining(emb_tr,y_tr_sem,num_tags_sem)
# inlp_syn = inlp_syn.to(device)
# P,P_is,Ws,Ps=inlp_syn.run_INLP_loop(20,dev_x=emb_dev,dev_y=y_dev_sem,min_acc=min_acc)
# print(f'the rank of P is :{np.linalg.matrix_rank(P)}')
# print(f'the rank gets removed :{768-np.linalg.matrix_rank(P)}')

# for index,m in enumerate(Ps):
# 	print(f'the rank of each iteration is :{np.linalg.matrix_rank(m)}')
# # for P_i in P_is:
# # 	print(np.linalg.matrix_rank(P_i))


# print('############# AFTER INLP #############')
# new_emb_tr = inlp_syn.embeddings
# P_t = torch.tensor(P)
# #print(torch.matrix_rank(P_t))
# if layerwise == -1:
# 	save_path_inlp = '../data/pmb_'+args.dataset+'/'+args.task+'_space_removed.pt'
# else:
# 	save_path_inlp = '../data/pmb_'+args.dataset+'/'+args.task+'_space_removed'+'_'+str(layerwise)+'.pt'
# torch.save(P_t,save_path_inlp)
# print('tensor types')
# print(P_t.type())
# print(emb_dev.type())
# P_t = P_t.type(torch.float)
# new_emb_dev = torch.matmul(P_t,emb_dev.T).T
# t_after = LinearClassifier(new_emb_tr,y_tr_syn,num_tags_syn)
# print('number of tags syntax?')
# print(num_tags_syn)
# print('number of unique syntax things?')
# print(np.unique(y_dev_syn.numpy()))
# t_after.optimize()
# print('eval result')
# t_after.eval(new_emb_dev,y_dev_syn)


print('### explore whether the projection matrices are all the same ###')
for i in range(1, 7):
	print('calculating the cosine similarity between the projection matrix of layer {}'.format(i))
	P_i = torch.load('../data/pmb_'+args.dataset+'/'+args.task+'_space_removed'+'_'+str(i)+'.pt')

print('### use one projection matrix on another layer ###')
print('### currently using projection matrix for layer {} on layer {} ###'.format(projection_layer, investigate_layer))
P = torch.load('../data/pmb_'+args.dataset+'/'+args.task+'_space_removed'+'_'+str(projection_layer)+'.pt') 
P = P.type(torch.float)

gold_tr = DataProcessing(args.dataset, 'train')
gold_tr.bert_tokenize()
train_emb = gold_tr.get_bert_embeddings(load_option, layerwise=investigate_layer)
emb_tr, y_tr, num_tags = gold_tr.from_sents_to_words(args.task,train_emb)

gold_test = DataProcessing(args.dataset,'test')
gold_test.bert_tokenize()

test_emb = gold_test.get_bert_embeddings(load_option, layerwise=investigate_layer)
## all in cpu right now
emb_test, y_test, num_tags = gold_test.from_sents_to_words(args.task,test_emb)

new_emb_tr = torch.matmul(P,emb_test.T).T 
new_emb_test = torch.matmul(P,emb_test.T).T

t_after = LinearClassifier(new_emb_tr,y_tr,num_tags)	 

t_after.optimize()
print('eval result')
t_after.eval(new_emb_test,y_test_syn)

sys.exit()









## calling eval
#eval_t = EvalClassifier(emb,y,num_tags,dev_x=,dev_y=)
#eval_t.optimize()

	

# loading data from saved pickle files
train_df = pd.read_pickle('../data/pmb_gold/gold_train.pkl')
train_input = train_df['text'].tolist()

dev_df=pd.read_pickle('../data/pmb_gold/gold_dev.pkl')
dev_input=dev_df['text'].tolist()
## tokenization
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
train_tokens,train_seq,train_mask = bert_tokenization(train_input, tokenizer)
dev_tokens,dev_seq,dev_mask = bert_tokenization(dev_input,tokenizer)

## TODO: adding a functionality here to load or output directly
# #tokens = tokens.to(device)
# train_seq = train_seq.to(device)
# train_mask = train_mask.to(device)
# outputs = model(train_seq,attention_mask = train_mask)
train_output = torch.load('../data/pmb_gold/gold_train_embeddings.pt')
dev_output = torch.load('../data/pmb_gold/gold_dev_embeddings.pt')

train_embeddings,train_y = from_sents_to_words(train_df,'sem',train_tokens,train_output)
dev_embeddings,dev_y = from_sents_to_words(dev_df,'sem',dev_tokens,dev_output)

# l=train_y.tolist()
# import collections
# print(collections.Counter(l))
# print(len(l))
# sys.exit()
# print(train_embeddings.shape)
# print(train_y.shape)
# print(dev_embeddings.shape)
# print(dev_y.shape)

#print(dev_embeddings.dtype)
num_epochs = 100

input_dim = train_embeddings.shape[1] #should be 768

sem_test = LinearClassifier(train_embeddings,train_y,num_tags)
#print(sem_test.stat_dict())
# sem_test = sem_test.to('cuda')
# sem_test.optimize()
# sys.exit()
# sem_test.optimize()
# eval_test = EvalClassifier(train_embeddings,train_y,num_tags,dev_embeddings,dev_y)
# eval_test.optimize(batch_size=1000)

sys.exit()
# lin_class = LinearClassifier(train_embeddings,train_y,num_tags,dev_x=dev_embeddings,dev_y=dev_y)
# print(lin_class.dev_x.shape)
# print(num_tags)
# #print(lin_class.output.shape)
# bm=lin_class.optimize()
# sys.exit()

orig_embeddings = train_embeddings
Ws = []
rowspace_projections = []
INLP_iterations = 3 #arbitrary choice for now
min_accuracy = 0.0 #can tune this as well

#INLP loop
## TODO: whether should include autoregressive
for i in tqdm(range(INLP_iterations)):

	#y are the gold standard labels for pos or srl
	linear_model, accuracy = train_linear_classifier(embeddings, y, num_epochs, num_tags, input_dim) #train linear classifier
	print('accuracy:'+str(accuracy))
	if accuracy < min_accuracy:
		continue 
	W = linear_model.weight
	Ws.append(W)
	P_rowspace_wi = get_rowspace_projection(W.detach()) #projection to W's rowspace
	rowspace_projections.append(P_rowspace_wi)
	P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
	embeddings = torch.tensor(P.dot(embeddings.detach().T).T, dtype = torch.float32) #project the embeddings

final_projection_matrix = P
guarded_embeddings = embeddings
 
