import torch
from transformers import BertModel, BertTokenizer,

import logging

#logging.basicConfig(level-logging.INFO) #turn on detailed logging
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                          output_hidden_states = True, # Whether the model returns all hidden-states.
                                                                            )

#load dataset
from datasets import load_dataset
dataset = load_dataset("conll2012_ontonotesv5")

inputs = tokenizer(dataset, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state

#train the classifier on the tasks to return protected attributes

#use last_hidden_states and protected attributes as input for INLP loop




