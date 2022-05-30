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
#logging.basicConfig(level-logging.INFO) #turn on detailed logging

## defining GPU here

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

#load data and BERT embeddings as in load_bert.py
## defining tokenizer and bert model here
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
model = AutoModel.from_pretrained('bert-base-uncased')
model = model.to(device)

# loading data from saved pickle files
train_df = pd.read_pickle('../data/train.pkl')
train_input = train_df['words'].tolist()

seq_lens = [len(item) for item in train_input]
max_len = 75

train_pos = train_df['pos_tags'].tolist()

## For Lindsay: consider this as the possible input for the function
train_srl = train_df['srl_frames'].tolist()

## input for the model
tokens, train_seq,train_mask = bert_tokenization(train_input, tokenizer)
#UNCOMMENT FOR TESTING WITH FIRST FIVE SENTENCES
#train_seq = train_seq[:5] 
#train_mask = train_mask[:5] 
#train_pos = train_pos[:5]

outputs = model(train_seq,attention_mask = train_mask)
word_inds = [tokens.words(i) for i in range(tokens['input_ids'].shape[0])]

#UNCOMMENT FOR TESTING WITH FIRST FIVE SENTENCES
#word_inds = word_inds[:5]

#convert to list of embeddings
d_list = []
for i in range(len(word_inds)):
	d = defaultdict(list)
	for ind,emb in zip(word_inds[i], outputs['last_hidden_state'][i]):
		if isinstance(ind, int):
			d[ind].append(emb)
	d_list.append(d) 
embeddings = torch.stack([torch.mean(torch.stack(d_list[i][j],0),0) if len(d_list[i][j]) > 1 else d_list[i][j][0] for i in range(len(d_list)) for j in d_list[i].keys()], 0)
y = torch.tensor([item for sublist in train_pos for item in sublist], dtype=torch.long)
#(not doing srl yet)

#set variables for INLP loop
num_epochs = 3
num_tags = 49
input_dim = embeddings.shape[1] #should be 768
orig_embeddings = embeddings
Ws = []
rowspace_projections = []
INLP_iterations = 10 #arbitrary choice for now
min_accuracy = 0.0 #can tune this as well

#INLP loop
for i in tqdm(range(INLP_iterations)):
	#y are the gold standard labels for pos or srl
	linear_model, accuracy = train_linear_classifier(embeddings, y, num_epochs, num_tags, input_dim) #train linear classifier
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
 
