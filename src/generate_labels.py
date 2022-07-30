import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
import json
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit

df1 = pd.read_pickle('/home2/qg07/homework_test/575project/data/pmb_gold/gold_train.pkl')
df2 = pd.read_pickle('/home2/qg07/homework_test/575project/data/pmb_gold/gold_dev.pkl')
df3 = pd.read_pickle('/home2/qg07/homework_test/575project/data/pmb_gold/gold_test.pkl')

a = pd.concat([df1,df2,df3])
a = a.drop('index',axis=1)

a.to_pickle('../data/pmb_gold/gold_all.pkl')


#ss = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=0)

#print(a['semantics_tags'])
#sys.exit()
# ccg_tag_list = np.concatenate(a['ccg_tags'].tolist())
# le = preprocessing.LabelEncoder()
# p = le.fit_transform(ccg_tag_list)

# print(len(le.classes_))
# sys
# d = dict(zip(le.classes_,le.transform(le.classes_).tolist()))
# with open('../data/pmb_silver/syn_mapping.json','w') as f:
#     json.dump(d,f)
# #print(a)