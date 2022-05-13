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
#logging.basicConfig(level-logging.INFO) #turn on detailed logging


## defining tokenizer and bert model here
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
model = AutoModel.from_pretrained('bert-base-uncased')


def bert_tokenization(words,max_len=75):
    '''
    this is a tokenization function that takes in a list object 
    and return a bert input format
    '''
    preprocessed_list = words
    tokens = tokenizer(preprocessed_list,
                    max_length=max_len,
                    truncation=True,
                    is_split_into_words=True,
                    padding=True,
                    return_tensors='pt')
    
    input_seq = tokens['input_ids']
    input_mask = tokens['attention_mask']
    return input_seq, input_mask

## TODO: for Lindsay: please fill in the following function so that
## we can derive the correct form
def encoding_srl(srls:List[Dict]):
    '''
    This function is to encode srl so that it returns the encoded form
    of SRL
    '''
    return None

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
train_seq,train_mask = bert_tokenization(train_input)


#outputs = model(**inputs) #not sure when we need this

#train the classifier on the tasks to return protected attributes

#use last_hidden_states and protected attributes as input for INLP loop






