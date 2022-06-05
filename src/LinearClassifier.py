import torch
from sklearn.metrics import accuracy_score

class LinearClassifier(torch.nn.Module):
	def __init__(self, input_embeddings,output,tag_size):
		'''
		a linear classifier for probe
		input_embeddings : a tensor with size [batch_size,embed_dim]
		output : a tensor with size [batch_size,1]
		tag_size : number of classes
		'''
		super().__init__()
		self.embeddings = input_embeddings
		self.output = output
		self.linear = torch.nn.Linear(input_embeddings.shape[1], tag_size)
		# TODO: possible class weight in loss function
		self.loss_func = torch.nn.CrossEntropyLoss()
	def forward(self,embeddings):
		# embedding size = [batch_size, embed_dim]
		# output size = [batch_size, num_tags]
		fc = self.linear(embeddings)
		#TODO: maybe later for logistics regression, output should be one hot
		#output_probs = torch.nn.functional.softmax(fc,dim=1)
		return fc
	
	def optimize(self,lr=0.01,num_epochs=100):
		optimizer = torch.optim.SGD(self.linear.parameters(), lr = lr)
		best_predictions = None
		for epoch in range(num_epochs):
			# TODO: shuffle the index of embeddings
			preds = self.forward(self.embeddings)
			loss = self.loss_func(preds,self.output)
			optimizer.zero_grad()
			loss.backward(retain_graph=True)
			optimizer.step()
			print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
			# TODO: implement stopping criterion
			best_predictions= preds
		final_pred = torch.argmax(best_predictions,dim=1).numpy()
		final_out = self.output.numpy()
		print(f'accuracy score:{accuracy_score(final_out,final_pred):.4f}')
		


# class INLPTraining(LinearClassifier):
# 	def __init__(self,):

