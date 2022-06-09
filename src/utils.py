from typing import List

import numpy as np
import scipy
import scipy.linalg
import torch


# the following three methods are created by the original INLP authors
# found from https://github.com/shauli-ravfogel/nullspace_projection/blob/master/src/inlp-oop/inlp.py
def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """

    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T)  # orthogonal basis

    w_basis * np.sign(w_basis[0][0])  # handle sign ambiguity
    P_W = w_basis.dot(w_basis.T)  # orthogonal projection on W's rowspace

    return P_W


def get_projection_to_intersection_of_nullspaces(rowspace_projection_matrices: List[np.ndarray], input_dim: int):
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
    uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
    N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
    :param rowspace_projection_matrices: List[np.array], a list of rowspace projections
    :param input_dim: input dim
    """

    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    P = I - get_rowspace_projection(Q)

    return P


def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int):
    """
    the goal of this function is to perform INLP on a set of user-provided directions
    (instead of learning those directions).
    :param directions: list of vectors, as numpy arrays.
    :param input_dim: dimensionality of the vectors.
    """

    rowspace_projections = []

    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)

    return P


# the following method is found in the amnestic probing code base
# in https://github.com/yanaiela/amnesic_probing/blob/main/amnesic_probing/tasks/utils.py
# with some modification
def remove_random_directions(x: np.ndarray, num_of_directions: int) -> np.ndarray:
    dim = x.shape[1]
    # creating random directions (vectors) within the range of -0.5 : 0.5
    rand_directions = [np.random.rand(1, dim) - 0.5 for _ in range(num_of_directions)]

    # finding the null-space of random directions
    rand_direction_p = debias_by_specific_directions(rand_directions, dim)

    # and projecting the original data into that space (to remove random directions)
    x_rand_direction = rand_direction_p.dot(x.T).T
    return x_rand_direction


def bert_tokenization(words,tokenizer, max_len=32):
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
    return tokens, input_seq, input_mask

def train_linear_classifier(X, y, num_epochs, num_tags, input_dim):
	linear_model = torch.nn.Linear(input_dim,num_tags)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(linear_model.parameters(), lr = 0.01)
	for epoch in range(num_epochs):
		logits = linear_model(X)
		loss = criterion(logits,y) 
		optimizer.zero_grad()
		loss.backward(retain_graph=True)
		optimizer.step()
		print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
	predictions = logits.argmax(axis=1)
	accuracy = predictions.float().mean()
	return linear_model, accuracy


def embeddingComponentBreakdown(x,P_sem,P_syn):
    """
    Takes in and outputs numpy arrays
    x is the original BERT embeddings (768,1)
    P_sem is the projection matrix defined in the INLP loop for the semantic tagging task (768,768)
    P_syn is the projection matrix defined in the INLP loop for the syntactic tagging task (768,768)
    """
    no_sem_emb = P_sem.matmul(x)
    no_syn_emb = P_syn.matmul(x)
    sem_emb = x - no_sem_emb
    syn_emb = x - no_syn_emb
    syn_less_sem_emb = P_sem.matmul(syn_emb)
    sem_less_syn_emb = P_syn.matmul(sem_emb)
    syn_sem_emb = syn_emb.dot(sem_emb) * sem_emb/np.sqrt(sem_emb**2)
    sem_syn_emb = sem_emb.dot(syn_emb) * syn_emb/np.sqrt(syn_emb**2)

    return no_sem_emb, no_syn_emb, syn_less_sem_emb, sem_less_syn_emb, syn_sem_emb, sem_syn_emb