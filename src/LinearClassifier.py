import torch
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import scipy
from typing import List
import tqdm

class LinearClassifier(torch.nn.Module):
	def __init__(self, input_embeddings,output,tag_size):
		'''
		a linear classifier for probe
		input_embeddings : a tensor with size [batch_size,embed_dim]
		output : a tensor with size [batch_size]
		tag_size : number of classes
		dev_x: dev set for stopping criterion
		dev_y: dev label for stopping criterion
		'''
		super().__init__()
		self.embeddings = input_embeddings
		self.output = output
		self.linear = torch.nn.Linear(input_embeddings.shape[1], tag_size)
		# class weight performs really worse
		# cls_weight = compute_class_weight('balanced',classes=np.array(range(tag_size)),y=output.numpy())
		# cls_weight = torch.tensor(cls_weight,dtype=torch.float)
		self.loss_func = torch.nn.CrossEntropyLoss()
	
	def forward(self,embeddings):
		# embedding size = [batch_size, embed_dim]
		# output size = [batch_size,tag_size]
		fc = self.linear(embeddings)
		return fc

	
	def optimize(self,lr=0.01,num_epochs=100):
		optimizer = torch.optim.AdamW(self.linear.parameters(), lr = lr)
		best_predictions = None
		best_loss = float('inf')
		stop_count = 0
		for epoch in range(num_epochs):
			preds = self.forward(self.embeddings)
			loss = self.loss_func(preds,self.output)
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()
			#print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
			#implement stopping criterion
			if loss.item()<best_loss:
				best_loss=loss.item()
				best_model= self.linear
				best_predictions = preds
				stop_count=0
			else:
				if stop_count ==3:
					break
				else:
					stop_count+=1
		final_pred = torch.argmax(best_predictions,dim=1).numpy()
		#final_dev = torch.argmax(best_dev,dim=1).numpy()
		#final_out = output.numpy()
		#dev_out = self.dev_y.numpy()
		train_acc = accuracy_score(self.output.numpy(),final_pred)
		#print(f'train accuracy score:{train_acc:.4f}')
		#print(f'dev accuracy score:{accuracy_score(self.dev_y.numpy(),final_dev):.4f}')
			
		return best_model,train_acc



class INLPTraining(LinearClassifier):
	def __init__(self,input_embeddings,output,tag_size):
		super().__init__(input_embeddings,output,tag_size)
		self.input_dim = self.embeddings.shape[1]
		# only used for the applying projection
		self.original_embedding = input_embeddings.T
	
	def get_rowspace_projection(self,model_weight):
		W = model_weight
		if np.allclose(W, 0):
			w_basis = np.zeros_like(W.T)
		else:
			w_basis = scipy.linalg.orth(W.T) # orthogonal basis 
		w_basis = w_basis * np.sign(w_basis[0][0]) # handle sign ambiguity
		P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace
		return P_W
	
	def get_projection_to_intersection_of_nullspaces(self, rowspace_projection_matrices: List[np.ndarray]):
		"""
		Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
		this function calculates the projection to the intersection of all nullspasces of the matrices w_1, ..., w_n.
		uses the intersection-projection formula of Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
		N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))
		:param rowspace_projection_matrices: List[np.array], a list of rowspace projections
		:param input_dim: input dim
		"""
		# This is werid because rowspace is not null space

		I = np.eye(self.input_dim)
		Q = np.sum(rowspace_projection_matrices, axis=0)
		P = I - self.get_rowspace_projection(Q)
		return P

	def reinitialize_classifier(self):
		in_size = self.linear.in_features
		out_size = self.linear.out_features
		self.linear = torch.nn.Linear(in_size,out_size)
	
	def apply_projection(self,P):
		'''
		applying projection of P to the embedding vectors
		'''
		P = torch.tensor(P,dtype=torch.float)
		self.embeddings =  torch.matmul(P,self.original_embedding).T

	def run_INLP_loop(self,iteration,min_acc=0.0):
		I = np.eye(self.input_dim)
		P = I
		Ws = []
		rowspace_projections = []
		for i in range(iteration):
			self.reinitialize_classifier()
			bm,acc=self.optimize()
			print(f'train acc for round {i} is {acc:.4f}')
			if acc < min_acc:
				# TODO: not sure it should be continue here
				continue
			W = bm.weight.detach().numpy()
			Ws.append(W)
			# Noted this is the projection space for W, not the null space
			P_rowspace_wi = self.get_rowspace_projection(W)
			rowspace_projections.append(P_rowspace_wi)
			# This line is supposed to get the null space for the projection space of W
			# Intuitively I think the rank makes sense, but I don't know how this will hold
			P_Nwi = I - P_rowspace_wi
			# This line is what they showed originally but the function looks weird
			#P = self.get_projection_to_intersection_of_nullspaces(rowspace_projections)
			P = np.matmul(P_Nwi,P)
			self.apply_projection(P)
		
		return P, rowspace_projections, Ws




