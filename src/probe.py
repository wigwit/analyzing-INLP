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

from load_bert import DataProcessing
from LinearClassifier import LinearClassifier, INLPTraining
from eval_classifier import EvalClassifier
#logging.basicConfig(level-logging.INFO) #turn on detailed logging

## defining GPU here
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

gold_train = DataProcessing('gold','train')
gold_train.bert_tokenize()
train_emb = gold_train.get_bert_embeddings('load')

emb_tr, y_tr, num_tags = gold_train.from_sents_to_words('sem',train_emb)

gold_dev = DataProcessing('gold','dev')
gold_dev.bert_tokenize()
dev_emb = gold_dev.get_bert_embeddings('load')

emb_dev, y_dev, num_tags = gold_dev.from_sents_to_words('sem',dev_emb)

# print(emb.is_cuda)
# print(y.is_cuda)
print(num_tags)
## TODO: weird inconsistency
eval_gold = EvalClassifier(emb_tr,y_tr,num_tags,emb_dev,y_dev)
eval_gold.optimize()

sys.exit()


## calling INLP
inlp_syn = INLPTraining(emb,y,num_tags)
inlp_syn = inlp_syn.to(device)
P,P_is,Ws=inlp_syn.run_INLP_loop(10)
print(f'the rank of P is :{np.linalg.matrix_rank(P)}')
for P_i in P_is:
	print(np.linalg.matrix_rank(P_i))

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
 
