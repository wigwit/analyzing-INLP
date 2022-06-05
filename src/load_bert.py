from re import M
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
from utils import bert_tokenization
#from nltk.stem import WordNetLemmatizer
#logging.basicConfig(level-logging.INFO) #turn on detailed logging

## defining GPU here

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

## defining tokenizer and bert model here
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
model = AutoModel.from_pretrained('bert-base-uncased')
model = model.to(device)




## TODO: for Lindsay: please fill in the following function so that
## we can derive the correct form

#wnl = WordNetLemmatizer()

def encoding_srl(srls:List[Dict], srl_ref=None):
    '''
    This function returns the encoded form of the SRL categories
    and an ordered dictionary for reference. The encoded SRL is a
    list where each entry contains a list of the SRL integer values 
    for the corresponding word. srl_ref is an ordered dictionary 
    whose keys correspond to SRL categories, which can be identified 
    by the integers corresponding to their positions/values. For 
    evaluation purposes we can provide a pre-defined srl_ref; all 
    SRL categories not seen during training should be categorized
    as 'unk'.
    '''
    train_ref = False
    if srl_ref is None: 
        srl_ref = {'unk':0}
        srl_ind = 1
        train_ref = True
    res = []
    for sent in srls:
        sent_res = [[] for i in range(len(sent[0]['frames']))]
        for i in range(len(sent)):
            vb = sent[i]['verb']
            #vb = wnl.lemmatize(vb,'v')
            for j in range(len(sent_res)):
                if sent[i]['frames'][j] != 'O':
                    fr = sent[i]['frames'][j][2:]
                    srl_key = vb + "_" + fr
                    if srl_key in srl_ref.keys():
                        n = srl_ref[srl_key]
                    elif train_ref:
                        srl_ref[srl_key] = srl_ind
                        n = srl_ind
                        srl_ind += 1
                    else:
                        n = 0
                    sent_res[j].append(n)
        res.append(sent_res)
    return srl_ref, res

# loading data from saved pickle files
train_df = pd.read_pickle('../data/pmb_gold/gold_train.pkl')
train_input = train_df['text'].tolist()

seq_lens = [len(item) for item in train_input]
## after doing some digging I decide to set max_len to be 75 to save computation
## This is because the total instances = 66582 and filtered instances = 66364
max_len = 32
# filtered = [i for i in seq_lens if i <=75]
# print(len(seq_lens))
# print(len(filtered))

train_ccg = train_df['ccg_tags'].tolist()

train_st = train_df['semantics_tags'].tolist()

## input for the model
tokens, train_seq,train_mask = bert_tokenization(train_input, tokenizer,max_len=32)
# print(train_seq,train_mask)
# print(train_input[30])
# detok = tokenizer.decode(train_seq[0])
# print(detok)
# sys.exit()
train_seq = train_seq.to(device)
train_mask = train_mask.to(device)

# freeze all the parameters
for param in model.parameters():
    param.requires_grad = False

model.eval()

with torch.no_grad():
    outputs = model(train_seq,attention_mask=train_mask)


train_output = outputs[0].detach().cpu()
# print(train_output[0])
# print(train_output[0].shape)
# print(train_input[0])
# print(train_mask[0])

# sys.exit()

torch.save(train_output,'../data/pmb_gold/gold_train_embeddings.pt')


