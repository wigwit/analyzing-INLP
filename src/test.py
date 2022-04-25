from datasets import load_dataset

dataset=load_dataset("conll2012_ontonotesv5","english_v12")
print(dataset['train'].features['sentences'][0]['named_entities'])