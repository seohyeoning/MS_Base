import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from Helpers.Variables import FILENAME_RES, FILENAME_HIST, FILENAME_HISTSUM, METRICS

"""
!! test가 없이 validation best score로 결과 내는 것 같다?! 확인
loss, model(baseline, ours),optimizer... : args.로 나중에 정리
"""
class Trainer():
    def __init__(self, args, model, MODEL_PATH):
        self.args = args
        self.model = model
        self.MODEL_PATH = MODEL_PATH
        self.lossfn = nn.MSELoss()
        self.set_optimizer()
        self.writer = SummaryWriter() # tensorboard, log directory

    def set_optimizer(self):
        if self.args.optimizer=='Adam':
            adam = torch.optim.Adam(self.model.parameters(), lr=float(self.args.lr))
            self.optimizer = adam
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, min_lr=1e-4, patience=0, verbose=1) # original patience=0, F1 score -> max
        elif self.args.optimizer=='AdamW':
            adamw = torch.optim.AdamW(self.model.parameters(), lr=float(self.args.lr))
            self.optimizer = adamw
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, min_lr=1e-4, patience=0, verbose=1) # original patience=0

    def train(self, train_loader, valid_loader):
        best_score = -np.inf
        trigger_time=0
        patience = 5
        verbose = True
        history = pd.DataFrame() # For saving metric history

        for epoch_idx in range(0, int(self.args.EPOCH)):
            result = self.train_epoch(train_loader, epoch_idx) # train
            history = pd.concat([history, pd.DataFrame(result).T], axis=0, ignore_index=True) 

            val_result = self.eval("valid", valid_loader, epoch_idx)
            valid_score = val_result        

            """Early Stopping by Comparing Loss"""
            if valid_score >= best_score: 
                if valid_score < 0: # valid_score = val_loss
                    self.scheduler.step(-val_result)
                    if verbose:
                        print(f'Validation loss decreased ({-best_score:.5f} --> {-valid_score:.5f}).  Saving model ...')
                else: # valid_score = val_f1 or others
                    self.scheduler.step(val_result)
                    if verbose:
                        print(f'Validation F1 Score increased ({best_score:.5f} --> {valid_score:.5f}).  Saving model ...')
                best_score = valid_score
                trigger_time=0
                torch.save(self.model.state_dict(), self.MODEL_PATH)
            else:
                trigger_time += 1
                if verbose:
                    print(f'EarlyStopping counting: {trigger_time} out of {patience}')
                if trigger_time > patience:
                    print(f'Early Stopped: epoch {epoch_idx} out of {int(self.args.EPOCH)}')
                    break
            print('')  

        # Load best performance model
        self.model.load_state_dict(torch.load(self.MODEL_PATH))
        
        history.columns = METRICS
        return history         

    """ Train """
    def train_epoch(self, train_loader, epoch=0):
        """
            x_train: (N, Ch, Seq)
            y_train: (N, classes)
            each element -> numpy array
        """
        self.model.train()
        lossfn=self.lossfn
        
        preds=[]
        targets=[]
        
        for datas in train_loader:
            data, target = datas['data'], datas['label']
            self.optimizer.zero_grad()
            output=self.model(data)

            loss = lossfn(output,target)
            loss.backward()
            self.optimizer.step()
            
            pred = output.argmax(dim=1)
            target = target.argmax(dim=1)
            
            preds.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            
        loss = loss.item()
        acc = accuracy_score(targets, preds)
        bacc=balanced_accuracy_score(targets, preds)
        f1=f1_score(targets, preds, average='macro')
        preci=precision_score(targets, preds, average='macro', zero_division=0)
        recall=recall_score(targets, preds, average='macro', zero_division=0)
        
        
        print('<train> epoch {} -- Loss: {:.5f}, Accuracy: {:.5f}%, Balanced Accuracy: {:.5f}%, f1score: {:.5f}, precision: {:.5f}, recall: {:.5f}'
            .format(epoch, loss, acc*100, bacc*100, f1, preci, recall))
        
        if (epoch%10==0) or (epoch==int(self.args.EPOCH)):
            self.write_tensorboard('train', epoch, loss, acc, bacc, f1, preci, recall)
        
        return [loss, acc, bacc, f1, preci, recall]

    """ EVALUATE """
    def eval(self, phase, loader, epoch=0):
        self.model.eval() ## evaluation mode로 변경 (dropout 사용 중지)
        lossfn = self.lossfn
        test_history = pd.DataFrame()
        test_loss = []
        preds=[]
        targets=[]
        
        with torch.no_grad(): # .eval함수와 torch.no_grad함수를 같이 사용하는 경향
            for datas in loader:
                data, target = datas['data'], datas['label']
                        
                outputs=self.model(data)
                test_loss.append(lossfn(outputs, target).item()) # sum up batch loss
                pred = outputs.argmax(dim=1,keepdim=False)# get the index of the max probability
                target = target.argmax(dim=1,keepdim=False)

                preds.extend(pred.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        loss = sum(test_loss)/len(loader.dataset)
        acc=accuracy_score(targets, preds)
        bacc=balanced_accuracy_score(targets,preds)
        f1=f1_score(targets,preds, average='macro', zero_division=0)
        preci=precision_score(targets,preds, average='macro', zero_division=0)
        recall=recall_score(targets,preds, average='macro', zero_division=0)

        print('<{}> epoch {} -- Loss: {:.5f}, Accuracy: {:.5f}%, Balanced Accuracy: {:.5f}%, f1score: {:.5f}, precision: {:.5f}, recall: {:.5f}'
            .format(phase, epoch, loss, acc*100, bacc*100, f1, preci, recall))

        self.write_tensorboard(phase, epoch, loss, acc, bacc, f1, preci, recall)
        
        if phase=="valid":
            return -loss # f1 or -loss
        elif phase=="test":
            result = [loss, acc, bacc, f1, preci, recall]
            test_history = pd.concat([test_history, pd.DataFrame(result).T], axis=0, ignore_index=True)
            test_history.columns = METRICS
            return test_history
        
    def save_result(self, tr_history, ts_history, res_dir):
        # save test history to csv
        res_path = os.path.join(res_dir, FILENAME_RES)
        ts_history.to_csv(res_path)
        print('Evaluation result saved')
        
        # save train history to csv
        hist_path = os.path.join(res_dir, FILENAME_HIST)
        histsum_path = os.path.join(res_dir, FILENAME_HISTSUM)
        tr_history.to_csv(hist_path)
        tr_history.describe().to_csv(histsum_path)
        print('History & History summary result saved')
        print('Tensorboard ==> \"tensorboard --logdir=runs\" \n')

    def write_tensorboard(self, phase, epoch, loss=0, acc=0, bacc=0, f1=0, preci=0, recall=0):
            if phase=='train':
                self.writer.add_scalar(f'{phase}/loss', loss, epoch)
                self.writer.add_scalar(f'{phase}/acc', acc, epoch)
            else:
                self.writer.add_scalar(f'{phase}/loss', loss, epoch)
                self.writer.add_scalar(f'{phase}/acc', acc, epoch)
                self.writer.add_scalar(f'{phase}/balanced_acc', bacc, epoch)  
                self.writer.add_scalar(f'{phase}/f1score', f1, epoch)
                self.writer.add_scalar(f'{phase}/precision', preci, epoch)
                self.writer.add_scalar(f'{phase}/recall', recall, epoch)     
    
