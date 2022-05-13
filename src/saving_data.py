from datasets import load_dataset
import pandas as pd
import itertools
import pickle

def from_ds_to_df(ds,filename=None):
    '''
    This functions takes in the dataset object with only 'sentences' and returns
    a pd.DataFrame object that removes empty srl_frames
    '''
    ## ds is a list of list, total set concatenate all the list together
    ## might want to change later ? since bert is training on A B sents
    total_set = list(itertools.chain(*ds))
    df = pd.DataFrame(total_set)
    filtered_df = df[df['srl_frames'].str.len()>0]
    if filename is not None:
        filtered_df.to_pickle(filename)
    return filtered_df


#load datasets
#check if the split is consistent
train_data = load_dataset("conll2012_ontonotesv5",'english_v4')['train']['sentences']
dev_data = load_dataset("conll2012_ontonotesv5",'english_v4')['validation']['sentences']
test_data = load_dataset("conll2012_ontonotesv5",'english_v4')['test']['sentences']

train_df = from_ds_to_df(train_data,'../data/train.pkl')
dev_df =from_ds_to_df(dev_data,'../data/dev.pkl')
test_df = from_ds_to_df(test_data,'../data/test.pkl')