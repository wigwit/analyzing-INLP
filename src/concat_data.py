import pandas as pd
import pickle as pkl

with open('../data/pmb_gold/gold_train.pkl', 'rb') as file:
    gold_train = pkl.load(file)

with open('../data/pmb_silver/silver_train.pkl', 'rb') as file:
    silver_train = pkl.load(file)

with open('../data/pmb_gold/gold_dev.pkl', 'rb') as file:
    gold_dev = pkl.load(file)

with open('../data/pmb_silver/silver_dev.pkl', 'rb') as file:
    silver_dev = pkl.load(file)

with open('../data/pmb_gold/gold_test.pkl', 'rb') as file:
    gold_test = pkl.load(file)

with open('../data/pmb_silver/silver_test.pkl', 'rb') as file:
    silver_test = pkl.load(file)


train_df = pd.concat([gold_train, silver_train])
train_df.to_pickle('../data/train.pkl')
dev_df = pd.concat([gold_dev, silver_dev])
dev_df.to_pickle('../data/dev.pkl')
test_df = pd.concat([gold_test, silver_test])
test_df.to_pickle('../data/test.pkl')

concat_df = pd.concat([train_df, dev_df, test_df])
concat_df.to_pickle('../data/concat.pkl')
