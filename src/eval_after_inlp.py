import torch
import numpy as np

from load_bert import DataProcessing
from LinearClassifier import LinearClassifier, INLPTraining
from eval_classifier import EvalClassifier

from utils import embeddingComponentBreakdown,remove_random_directions

P_sem = torch.load('sem_space_removed.pt')
P_syn = torch.load('syn_space_removed.pt')

gold_train = DataProcessing('gold','train')
gold_train.bert_tokenize()
train_emb = gold_train.get_bert_embeddings('load')

emb_tr, y_tr_syn,num_tags_syn = gold_train.from_sents_to_words('syn',train_emb)


gold_dev = DataProcessing('gold','test')
gold_dev.bert_tokenize()
dev_emb = gold_dev.get_bert_embeddings('load')

emb_dev, y_dev_syn, dum = gold_dev.from_sents_to_words('syn',dev_emb)

# print(P_sem.shape)
# print(P_syn.shape)
new_emb_trs = embeddingComponentBreakdown(emb_tr.T.double(),P_sem,P_syn)
new_emb_devs = embeddingComponentBreakdown(emb_dev.T.double(),P_sem,P_syn)
sim =torch.nn.functional.cosine_similarity(new_emb_trs[1],new_emb_trs[0])
print(sum(sim)/len(sim))

sys.exit()
# print(new_emb_trs[4].shape)
# print(768-torch.matrix_rank(new_emb_trs[4]))

# print(new_emb_trs[2].shape)

# new_emb_tr = torch.matmul(P,emb_tr.T).T
# new_emb_dev = torch.matmul(P,emb_dev.T).T

# new_emb_tr = remove_random_directions(emb_tr.numpy(),77)
# new_emb_dev = remove_random_directions(emb_dev.numpy(),77)

# new_emb_tr = torch.tensor(new_emb_tr)
# new_emb_dev = torch.tensor(new_emb_dev)

t_after = LinearClassifier(new_emb_trs[4],y_tr_syn,num_tags_syn)
t_after.optimize()
t_after.eval(new_emb_devs[4],y_dev_syn)


# print(768-torch.matrix_rank(P_sem))
# print(768-torch.matrix_rank(P_syn))