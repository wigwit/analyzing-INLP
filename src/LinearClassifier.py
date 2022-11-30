import torch
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import scipy
from typing import List
import tqdm
import random

#random.seed(42)

## defining GPU here
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

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
		random.seed(42)
		## everything defined in GPU
		self.embeddings = input_embeddings.double()
		self.output = output
		self.linear = torch.nn.Linear(input_embeddings.shape[1], tag_size,device=device,dtype=torch.double)
		# class weight performs really worse
		# cls_weight = compute_class_weight('balanced',classes=np.array(range(tag_size)),y=output.numpy())
		# cls_weight = torch.tensor(cls_weight,dtype=torch.float)
		self.loss_func = torch.nn.CrossEntropyLoss()
	
	def forward(self,embeddings):
		# embedding size = [batch_size, embed_dim]
		# output size = [batch_size,tag_size]
		emb = embeddings.to(device)
		emb = emb.double()
		fc = self.linear(emb)
		return fc

	def eval(self,dev_x,dev_y):
		with torch.no_grad():
			# TODO: batching if necessary
			dev_x = dev_x.to(device)
			dev_y = dev_y.to(device)
			dev_pred = self.forward(dev_x)
			loss = self.loss_func(dev_pred,dev_y)

		final_dev =  torch.argmax(dev_pred,dim=1).cpu().numpy()
		acc = accuracy_score(dev_y.cpu().numpy(),final_dev)
		print(f'dev accuracy score:{acc:.4f}')
		return dev_pred, loss.item(),acc

	
	def batched_input(self,*args,batch_size=64):
		data_set = TensorDataset(args[0],args[1])
		dataloader = DataLoader(data_set,batch_size=batch_size)
		return dataloader
	
	def optimize(self,lr=0.01,num_epochs=10):
		optimizer = torch.optim.AdamW(self.linear.parameters(), lr = lr)
		best_predictions = None
		best_loss = float('inf')
		stop_count = 0
		output = self.output.to(device)
		dataloader = self.batched_input(self.embeddings,output)
		for epoch in range(num_epochs):
			preds = []
			total_loss = 0
			for emb,label in dataloader:
				optimizer.zero_grad()
				pred = self.forward(emb)
				loss = self.loss_func(pred,label)
				loss.backward(retain_graph=True)
				optimizer.step()
				pred = pred.to('cpu')
				preds.append(pred)
				total_loss += loss.item()
			
			total_loss = total_loss/len(dataloader)
			preds = torch.cat(preds)
			#print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
			#implement stopping criterion
			if total_loss<best_loss:
				best_loss=total_loss
				best_model= self.linear
				best_predictions = preds
				stop_count=0
			else:
				if stop_count ==3:
					break
				else:
					stop_count+=1
		final_pred = torch.argmax(best_predictions,dim=1).cpu().numpy()
		#final_dev = torch.argmax(best_dev,dim=1).numpy()
		#final_out = output.numpy()
		#dev_out = self.dev_y.numpy()
		train_acc = accuracy_score(self.output.numpy(),final_pred)
		print(f'train accuracy score:{train_acc:.4f}')
		#print(f'dev accuracy score:{accuracy_score(self.dev_y.numpy(),final_dev):.4f}')
			
		return best_model,train_acc



class INLPTraining(LinearClassifier):
	def __init__(self,input_embeddings,output,tag_size):
		super().__init__(input_embeddings,output,tag_size)
		self.input_dim = self.embeddings.shape[1]
		# only used for the applying projection
		self.original_embedding = input_embeddings.T
	
	def get_rowspace_projection(self,model_weight):
		"""
		Defines the rowspace projection onto the vectorspace spanned by the columns of a 2D matrix.

		Parameters:
		-----------
		linear_matrix
			The 2D matrix used to perform classification in a linear classifier.

		Returns:
		--------
		rowspace_projection_matrix
			The 2D matrix that, when multiplied by an embedding, results in the projection of that embedding onto the
			vectorspace spanned by the columns of linear_matrix.
    	"""
		W = model_weight
		if np.allclose(W, 0):
			w_basis = np.zeros_like(W.T)
		else:
			w_basis = scipy.linalg.orth(W.T) # orthogonal basis 
		w_basis = w_basis * np.sign(w_basis[0][0]) # handle sign ambiguity
		P_W = w_basis.dot(w_basis.T) # orthogonal projection on W's rowspace
		return P_W
	
	def get_projection_to_intersection_of_nullspaces(self, input_dim,rowspace_projection_matrices: List[np.ndarray]):
		"""
		Determines the matrix that projects onto the intersection of the nullspaces of the provided
		rowspace_projection_matrices.

		Details:
		--------
		Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n), this function calculates the projection to
		the intersection of all nullspasces of the matrices w_1, ..., w_n. It uses the intersection-projection formula of
		Ben-Israel 2013 http://benisrael.net/BEN-ISRAEL-NOV-30-13.pdf:
		N(w1)∩ N(w2) ∩ ... ∩ N(wn) = N(P_R(w1) + P_R(w2) + ... + P_R(wn))

		Parameters:
		-----------
		input_dim: dimension of row space projection matrices
		rowspace_projection_matrices
			A list of matrices. Each matrix projects embeddings onto the rowspace of its respective linear classifier.

		Returns:
		--------
		nullspace_projection_matrix
			The 2D matrix that, when multiplied by an embedding, results in the projection of that embedding onto the
			intersection of the nullspaces of the provided rowspace projection matrices.
		"""
		# This is werid because Q is not normalized so the N(P) = I-P does not work
		I = np.eye(input_dim)
		Q = np.sum(rowspace_projection_matrices, axis=0)
		P = I - self.get_rowspace_projection(Q)
		return P

	def reinitialize_classifier(self):
		## may be empty cache here
		in_size = self.linear.in_features
		out_size = self.linear.out_features
		## this random seeding might be messing with the training
		## might remove it just for testing
		random.seed(42)
		self.linear = torch.nn.Linear(in_size,out_size,device=device,dtype=torch.double)
	
	def apply_projection(self,P):
		'''
		applying projection of P to the embedding vectors
		'''
		## may be empty cache here
		P = torch.tensor(P,dtype=torch.float)
		self.embeddings =  torch.matmul(P,self.original_embedding).T
		self.embeddings = self.embeddings.double()

	def run_INLP_loop(self,iteration,dev_x=None,dev_y=None,min_acc=0.0):
		I = np.eye(self.input_dim)
		P = I
		Ws = []
		all_P = []
		rowspace_projections = []
		for i in range(iteration):
			self.reinitialize_classifier()
			bm,acc=self.optimize()
			if dev_x is not None:
				dum1,dum2,acc=self.eval(dev_x,dev_y)
				print(f'dev acc for round {i} is {acc:.4f}')
			if acc < min_acc:
				# TODO: not sure it should be continue here
				break
			W = bm.weight.detach().cpu().numpy()
			Ws.append(W)
			# Noted this is the projection space for W, not the null space
			P_rowspace_wi = self.get_rowspace_projection(W)
			rowspace_projections.append(P_rowspace_wi)
			# This line is supposed to get the null space for the projection space of W
			# Intuitively I think the rank makes sense, but I don't know how this will hold
			P_Nwi = self.get_projection_to_intersection_of_nullspaces(input_dim=P_rowspace_wi.shape[0],
                                                         rowspace_projection_matrices=rowspace_projections)
			# This line is what they showed originally but the function looks weird
			#P = self.get_projection_to_intersection_of_nullspaces(rowspace_projections)
			P = np.matmul(P_Nwi,P)
			all_P.append(P)
			self.apply_projection(P)
		
		return P, rowspace_projections, Ws,all_P




