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


def from_sents_to_words(input_df,task_keyword,tokens,output):
	'''
	this function checks the tokenization process and transfer data into word level
	Arguments:
		input_df: a pd.DataFrame object
		task_keyword: {syn,sem}
		tokens: bert tokenized object
		output: a tensor with shape (sentence num, max_seq_len, embedding dim)
	Returns: 
		embeddings: a tensor with shape (word num, embedding dim)
		y: a tensor with shape (word num,1)
	'''
	input_text = input_df['text'].tolist()
	if task_keyword == 'syn':
		input_y = input_df['ccg_tags'].tolist()
	else:
		input_y = input_df['semantics_tags'].tolist()
	label_encoder = preprocessing.LabelEncoder()
	label_encoder.fit(np.concatenate(input_y))
	encoded_y = [label_encoder.transform(l) for l in input_y]
	## mapping tokens back to word ids
	word_inds = [tokens.word_ids(i) for i in range(len(input_y))]
	d_list = []
	embed_by_sent = []
	skip_ind = []
	for i in range(len(word_inds)):
		d = defaultdict(list)
		for ind,emb in zip(word_inds[i], output[i]):
			if isinstance(ind, int):
				d[ind].append(emb)
	
		word_in_sent = [torch.mean(torch.stack(d[k],0),0) if len(d[k])>1 else d[k][0] for k in d.keys()]
		if len(word_in_sent)!= len(input_text[i]):
			skip_ind.append(i)
			continue
		embed_by_sent.append(torch.stack(word_in_sent,0))
		d_list.append(d)
	embeddings = torch.cat(embed_by_sent)
	y = torch.tensor([item for i, sublist in enumerate(encoded_y) if i not in skip_ind for item in sublist ], dtype=torch.long)
	return embeddings,y


	

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

# print(train_embeddings.shape)
# print(train_y.shape)
# print(dev_embeddings.shape)
# print(dev_y.shape)

num_epochs = 100
num_tags = len(np.unique(train_y.numpy()))
input_dim = train_embeddings.shape[1] #should be 768


lin_class = LinearClassifier(train_embeddings,train_y,num_tags,dev_x=dev_embeddings,dev_y=dev_y)
print(lin_class.dev_x.shape)
print(num_tags)
#print(lin_class.output.shape)
bm=lin_class.optimize()
sys.exit()

orig_embeddings = train_embeddings
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
 
