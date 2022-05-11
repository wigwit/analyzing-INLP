import torch
from transformers import BertModel, BertTokenizer

import logging

#logging.basicConfig(level-logging.INFO) #turn on detailed logging
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                          output_hidden_states = True, # Whether the model returns all hidden-states.
                                                                            )

#load dataset
from datasets import load_dataset
train_data = load_dataset("conll2012_ontonotesv5",'english_v12')['train']
dev_data = load_dataset("conll2012_ontonotesv5",'english_v12')['validation']
test_data = load_dataset("conll2012_ontonotesv5",'english_v12')['test']

#encode data into tokenizable format

def encode(doc):
    return {'sentences': doc['sentences']} #fix this to form a dict of lists of strings
#also check how you did this last year
#and maybe meet with qingxia if you need

sentences = [sentence['words'] for doc in train_data for sentence in doc['sentences']]
inputs = tokenizer(sentences, return_tensors="pt")

#outputs = model(**inputs) #not sure when we need this

#train the classifier on the tasks to return protected attributes

#use last_hidden_states and protected attributes as input for INLP loop




