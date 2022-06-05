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
from collections import defaultdict
from utils import bert_tokenization, get_projection_to_intersection_of_nullspaces, get_rowspace_projection, train_linear_classifier
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
from LinearClassifier import LinearClassifier
#logging.basicConfig(level-logging.INFO) #turn on detailed logging

## defining GPU here

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

#load data and BERT embeddings as in load_bert.py
## defining tokenizer and bert model here
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
# model = AutoModel.from_pretrained('bert-base-uncased')
# model = model.to(device)

# loading data from saved pickle files
train_df = pd.read_pickle('../data/pmb_gold/gold_train.pkl')
#train_df =train_df[:1000]
train_input = train_df['text'].tolist()


train_ccg = train_df['ccg_tags'].tolist()

train_words=train_df['text'].tolist()
count_words= [len(l) for l in train_words]

ccg_encoder = preprocessing.LabelEncoder()
ccg_encoder.fit(np.concatenate(train_ccg))

ccg_output =[ ccg_encoder.transform(l) for l in train_ccg]

train_st = train_df['semantics_tags'].tolist()

# ## input for the model
tokens, train_seq,train_mask = bert_tokenization(train_input, tokenizer)
# #tokens = tokens.to(device)
# train_seq = train_seq.to(device)
# train_mask = train_mask.to(device)

# outputs = model(train_seq,attention_mask = train_mask)
output = torch.load('../data/pmb_gold/gold_train_embeddings.pt')
# output = output[:1000]
word_inds = [tokens.words(i) for i in range(train_seq.shape[0])]

#convert to list of embeddings
d_list = []
embed_by_sent = []
skip_ind = []
for i in range(len(word_inds)):
	d = defaultdict(list)
	for ind,emb in zip(word_inds[i], output[i]):
		if isinstance(ind, int):
			d[ind].append(emb)
	
	word_in_sent = [torch.mean(torch.stack(d[k],0),0) if len(d[k])>1 else d[k][0] for k in d.keys()]
	if len(word_in_sent)!= count_words[i]:
		skip_ind.append(i)
		continue
	embed_by_sent.append(torch.stack(word_in_sent,0))
	d_list.append(d) 
# 30k*768
embeddings = torch.cat(embed_by_sent)
# 30k*1
y = torch.tensor([item for i, sublist in enumerate(ccg_output) if i not in skip_ind for item in sublist ], dtype=torch.long)

#(not doing srl yet)
#print(count_words)
# print(embeddings.shape)
# print(y.shape)
# sys.exit()
#set variables for INLP loop
num_epochs = 100
num_tags = len(ccg_encoder.classes_)
input_dim = embeddings.shape[1] #should be 768
orig_embeddings = embeddings

lin_class = LinearClassifier(embeddings,y,num_tags)
lin_class.optimize()
lin_class.optimize()
lin_class.optimize()
sys.exit()

Ws = []
rowspace_projections = []
INLP_iterations = 3 #arbitrary choice for now
min_accuracy = 0.0 #can tune this as well

#INLP loop
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
 
