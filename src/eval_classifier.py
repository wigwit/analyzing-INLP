from LinearClassifier import LinearClassifier
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from sklearn.metrics import accuracy_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')


class EvalClassifier(torch.nn.Module):
    def __init__(self,input_embeddings,output,tag_size,hidden_dim,dev_x,dev_y):
        '''
        A Eval Classifier for eval task, inherited from Linear Classifer
        But With batching and different logistics regression
        '''
        super().__init__()
        self.embeddings = input_embeddings
        self.output = output
        self.dev_x = dev_x.to(device)
        self.dev_y = dev_y.to(device)
        self.linear = torch.nn.Linear(input_embeddings.shape[1], tag_size,device=device)
        #self.linear2 = torch.nn.Linear(hidden_dim,tag_size,device=device)
        self.loss_func = torch.nn.CrossEntropyLoss()
    
    def forward(self,embeddings):
        x = self.linear(embeddings)
        # x = torch.nn.functional.sigmoid(x)
        # out = self.linear2(x)
        
        return x
    
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
    
    def optimize(self,batch_size=24954, lr=0.01,num_epochs=100):
        optimizer = torch.optim.AdamW(self.linear.parameters(), lr = lr)
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
                emb = emb.to(device)
                label = label.to(device)
                pred = self.forward(emb)
                train_preds.append(pred)
                loss = self.loss_func(pred,label)
                self.linear.zero_grad()
                loss.backward(retain_graph=True)
                # clip the gradient
                #torch.nn.utils.clip_grad_norm_(self.linear.parameters(), 1.0)
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

        
