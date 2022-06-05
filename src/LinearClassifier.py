import torch
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class LinearClassifier(torch.nn.Module):
	def __init__(self, input_embeddings,output,tag_size,dev_x=None,dev_y=None):
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
		self.dev_x = dev_x
		self.dev_y = dev_y
	
	def forward(self,embeddings):
		# embedding size = [batch_size, embed_dim]
		# output size = [batch_size]
		fc = self.linear(embeddings)
		#TODO: maybe later for logistics regression, output should be one hot
		#output_probs = torch.nn.functional.softmax(fc,dim=1)
		return fc

	def eval(self):
		with torch.no_grad():
			dev_pred = self.linear(self.dev_x)
			loss = self.loss_func(dev_pred,self.dev_y)
		return dev_pred,loss.item()

	
	def optimize(self,lr=0.01,num_epochs=1000):
		optimizer = torch.optim.SGD(self.linear.parameters(), lr = lr)
		best_predictions = None
		best_loss = float('inf')
		stop_count = 0
		for epoch in range(num_epochs):
			preds = self.forward(self.embeddings)
			loss = self.loss_func(preds,self.output)
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()
			print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
			#implement stopping criterion
			dev_pred,dev_loss=self.eval()
			if dev_loss<best_loss:
				best_loss=dev_loss
				best_model= self.linear
				best_predictions = preds
				best_dev = dev_pred
				stop_count=0
			else:
				if stop_count ==100:
					break
				else:
					stop_count+=1
		final_pred = torch.argmax(best_predictions,dim=1).numpy()
		#dev_pred = best_model(self.dev_x)
		final_dev = torch.argmax(best_dev,dim=1).numpy()
		#final_dev = torch.argmax(best_dev,dim=1).numpy()
		#final_out = output.numpy()
		#dev_out = self.dev_y.numpy()
		print(f'train accuracy score:{accuracy_score(self.output.numpy(),final_pred):.4f}')
		print(f'dev accuracy score:{accuracy_score(self.dev_y.numpy(),final_dev):.4f}')
			
		return best_model

		


# class INLPTraining(LinearClassifier):
# 	def __init__(self,):

