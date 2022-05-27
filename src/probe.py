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
from utils import bert_tokenization, get_projection_to_intersection_of_nullspaces, get_rowspace_projection 
#logging.basicConfig(level-logging.INFO) #turn on detailed logging

## defining GPU here

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

## defining tokenizer and bert model here
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
model = AutoModel.from_pretrained('bert-base-uncased')
model = model.to(device)

# loading data from saved pickle files
train_df = pd.read_pickle('../data/train.pkl')
train_input = train_df['words'].tolist()

seq_lens = [len(item) for item in train_input]
## after doing some digging I decide to set max_len to be 75 to save computation
## This is because the total instances = 66582 and filtered instances = 66364
max_len = 75
# filtered = [i for i in seq_lens if i <=75]
# print(len(seq_lens))
# print(len(filtered))

train_pos = train_df['pos_tags'].tolist()

## For Lindsay: consider this as the possible input for the function
train_srl = train_df['srl_frames'].tolist()

## input for the model
tokens, train_seq,train_mask = bert_tokenization(train_input, tokenizer)

outputs = model(train_seq,attention_mask = train_mask)
word_inds = [tokens.words(i) for i in range(tokens['input_ids'].shape[0])]

#convert to list of embeddings
d_list = []
for i in range(len(word_inds)):
	d = defaultdict(list)
	for ind,emb in zip(word_inds[i], outputs['last_hidden_state'][i]):
		if isinstance(ind, int):
			d[ind].append(emb
	d_list.append(d) 
embeddings = torch.stack([torch.mean(torch.stack(d_list[i][j],0),0) if len(d_list[i][j]) > 1 else d_list[i][j][0] for i in range(len(d_list)) for j in d_list[i].keys()], 0)
y = torch.FloatTensor([item for sublist in train_pos for item in sublist])

#train the linear classifier
num_tags = 49 
input_dim = 768
linear_model = torch.nn.Linear(input_dim,num_tags)
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(linear_model.parameters(), lr = 0.01)
for epoch in range(num_epochs):
	y_pred = linear_model(embeddings)
	loss = criterion(y_pred,y) 
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

#calculate nullspace
#find orthogonal projection matrix
#do the projection

#null space projection:
def run_inlp(linear_model):
	#returns projection matrix
	Ws = []
	rowspace_projections = []
	W = linear_model.get_weights()
	Ws.append(W)
	P_rowspace_wi = get_rowspace_projection(W) # projection to W's rowspace
	rowspace_projections.append(P_rowspace_wi)
	P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
	return P

#so how do we et the guarded data then?
