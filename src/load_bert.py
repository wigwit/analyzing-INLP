import torch
from transformers import AutoTokenizer, AutoModel
import sys
import logging
from datasets import load_dataset
import pandas as pd
import itertools
import pdb

#logging.basicConfig(level-logging.INFO) #turn on detailed logging


## defining tokenizer and bert model here
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
model = AutoModel.from_pretrained('bert-base-uncased')
# for now i disable the hidden states function since we are only using bert to probe                                    
# output_hidden_states = True, # Whether the model returns all hidden-states)



def bert_tokenization(words):
    '''
    this is a tokenization function that takes in a df.series object 
    and return a bert input format
    '''
    preprocessed_list = words.tolist()
    # this line of code is little buggy, will come back later
    tokens = tokenizer.batch_encode_plus(preprocessed_list,padding=True)
    input_seq = torch.tensor(tokens['input_ids'])
    input_mask = torch.tensor(tokens['attention_mask'])
    return input_seq, input_mask


# loading data from saved csv files
train_df = pd.read_csv('../data/train.csv')
train_input = train_df['words']
train_pos = train_df['pos_tags']
train_seq,train_mask = bert_tokenization(train_input)



# def words_of(dataset):
#     return {'sentences': doc['sentences']} #fix this to form a dict of lists of strings
#also check how you did this last year
# and maybe meet with qingxia if you need
# sentences = [sentence['words'] for doc in train_data for sentence in doc['sentences']]


# inputs = tokenizer(sentences, return_tensors="pt")

#outputs = model(**inputs) #not sure when we need this

#train the classifier on the tasks to return protected attributes

#use last_hidden_states and protected attributes as input for INLP loop






