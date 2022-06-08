from LinearClassifier import LinearClassifier
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.metrics import accuracy_score

class EvalClassifier(LinearClassifier):
    def __init__(self,input_embeddings,output,tag_size,dev_x,dev_y):
        '''
        A Eval Classifier for eval task, inherited from Linear Classifer
        But With batching and different logistics regression
        '''
        super().__init__(input_embeddings,output,tag_size)
        self.dev_x = dev_x
        self.dev_y = dev_y
        # TODO: tranfer output to one hot vector
        #self.output_vecs = 
        #self.loss_func = torch.nn.NLLLoss()
    
    def forward(self,embeddings):
        fc = self.linear(embeddings)
        #output_probs = torch.nn.functional.softmax(fc,dim=1)
        return fc
    
    def batching_data(self,dset='train',batch_size=32):
        if dset == 'train':
            data = TensorDataset(self.embeddings,self.output)
        elif dset == 'dev':
            data = TensorDataset(self.dev_x,self.dev_y)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data,sampler=sampler,batch_size=batch_size)
        return dataloader
    
    def eval(self):
        with torch.no_grad():
            # TODO: batching if necessary
            dev_pred = self.forward(self.dev_x)
            loss = self.loss_func(dev_pred,self.dev_y)
        return dev_pred, loss.item()
    
    def optimize(self,batch_size=32, lr=0.01,num_epochs=12):
        optimizer = torch.optim.Adagrad(self.linear.parameters(), lr = lr)
        best_predictions_tr = None
        best_predictions_dv = None
        best_loss=float('inf')
        stop_count = 0
        for epoch in range(num_epochs):
            train_batch = self.batching_data(batch_size=batch_size)
            train_loss = 0
            train_preds = []
            dev_preds = []
            for emb, label in train_batch:
                pred = self.forward(emb)
                train_preds.append(pred)
                loss = self.loss_func(pred,label)
                self.linear.zero_grad()
                loss.backward(retain_graph=True)
                # clip the gradient
                torch.nn.utils.clip_grad_norm_(self.linear.parameters(), 1.0)
                optimizer.step()
                train_loss+=loss.item()
            avg_loss = train_loss/len(train_batch)
            print(f'epoch: {epoch+1}, train loss = {avg_loss:.4f}')

            dev_preds,dev_loss = self.eval()
            print(f'epoch: {epoch+1}, dev loss = {dev_loss:.4f}')
            if dev_loss<best_loss:
                # train_preds = torch.cat(train_preds)
                # dev_preds = torch.cat(dev_preds)
                best_loss = dev_loss
                best_predictions_tr = torch.cat(train_preds)
                best_predictions_dv = dev_preds
                best_model=self.linear
                stop_count=0
            else:
                if stop_count == 3:
                    break
                else:
                    stop_count+=1
            
        final_train = torch.argmax(best_predictions_tr,dim=1).cpu().numpy()
        final_dev = torch.argmax(best_predictions_dv,dim=1).cpu().numpy()
        print(f'train accuracy score:{accuracy_score(self.output.cpu().numpy(),final_train):.4f}')
        print(f'dev accuracy score:{accuracy_score(self.dev_y.cpu().numpy(),final_dev):.4f}')

        
