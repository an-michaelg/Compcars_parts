# wrapper module for compcars training

import os
import yaml

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from tqdm import tqdm
import wandb

from get_model import GatedAttentionClassifier, AveragingClassifier
from compcars_dataloader import get_car_views, get_car_types

class Network(object):
    """Wrapper for training and testing pipelines."""

    def __init__(self, config, classind_to_model, taxonomy):
        """Initialize configuration."""
        super().__init__()
        #torch.autograd.set_detect_anomaly(True)
        
        self.config = config
        self.epsilon = 1e-8
        # track some consistent information about the dataset
        self.car_views = get_car_views()
        self.car_types = get_car_types()
        self.car_models = classind_to_model
        self.num_classes = len(self.car_models)
        print("Number of classes: {}".format(self.num_classes))
        self.taxonomy = taxonomy
        
        # instantiate the architecture
        # self.embedding_dim = config['embedding_dim']
        # self.dense_hidden_dim = config['dense_hidden_dim']
        # self.pretrained = config['encoder_pretrained']
        # self.use_gated = config['use_gated']
        self.model = AveragingClassifier(config['embedding_dim'], 
                                              config['dense_hidden_dim'],
                                              self.num_classes,
                                              config['encoder_pretrained'])
        
        if config['use_cuda']:
            if torch.cuda.is_available():
                self.use_cuda = True
                print('Using GPU acceleration: GPU{}'.format(torch.cuda.current_device()))
            else:
                self.use_cuda = False
                print('Warning: CUDA is requested but not available. Using CPU')
        else:
            self.use_cuda = False
            print('Using CPU')
            
        if self.use_cuda:
            self.model = self.model.cuda()
            
        self.optimizer, self.scheduler = self.configure_optimizers()

        # init auxiliary stuff such as log_func
        self._init_aux()

    def _init_aux(self):
        """Intialize aux functions, features."""
        # Define func for logging.
        self.log_func = print

        # Define directory where we save states such as trained model.
        if self.config['mode'] == 'test':
            # find the location of the weights and use the same directory as the log directory
            if self.config['model_load_dir'] is None:
                raise AttributeError('For test-only mode, please specify the model state_dict folder')
            self.log_dir = os.path.join(self.config['log_dir'], self.config['model_load_dir'])
        else: # training
            if self.config['use_wandb']:
                # use the wandb folder name as the new directory
                self.log_dir = os.path.join(self.config['log_dir'], wandb.run.name)
            else:
                # use the base folder as the save folder
                self.log_dir = self.config['log_dir']
        # if self.config['use_wandb']:
        #     if self.config['mode'] == 'test':
        #         # use the previous directory as the log directory
        #         if self.config['model_load_dir'] is None:
        #             raise AttributeError('For test-only mode, please specify the model state_dict folder')
        #         self.log_dir = os.path.join(self.config['log_dir'], self.config['model_load_dir'])
        #     else:
        #         # after this switch to the new log directory
        #         self.log_dir = os.path.join(self.config['log_dir'], wandb.run.name)
        # else:
        #     self.log_dir = self.config['log_dir']
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # File that we save trained model into.
        self.checkpts_file = os.path.join(self.log_dir, "checkpoint.pth")

        # We save model that achieves the best performance: early stopping strategy.
        self.bestmodel_file = os.path.join(self.log_dir, "best_model.pth")
            
        # We can save a copy of the training configurations for future reference
        if self.config['mode'] != 'test':
            yaml_filename = os.path.join(self.log_dir, "config.yaml")
            with open(yaml_filename, 'w') as file:
                yaml.dump(self.config, file)
    
    def _save(self, pt_file):
        """Saving trained model."""

        # Save the trained model.
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            pt_file,
        )

    def _restore(self, pt_file):
        """Restoring trained model."""
        print(f"restoring {pt_file}")

        # Read checkpoint file.
        load_res = torch.load(pt_file)
        # Loading model.
        self.model.load_state_dict(load_res["model"])
        self.optimizer.load_state_dict(load_res["optimizer"])
            
    def configure_optimizers(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=self.config['lr'])
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     self.optimizer,T_max=self.config['num_epochs'])
        vb = (self.config['mode'] != 'test')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.config['exp_gamma'], verbose=vb)
        return optimizer, scheduler

    def _get_network_output(self, data):
        imgs = data[0]
        targets = data[1]
        # Transfer data from CPU to GPU.
        if self.use_cuda:
            imgs = [i.cuda() for i in imgs]
            targets = targets.cuda()
        
        # get the logit output
        logits = self.model(imgs)
        return imgs, targets, logits
                
    def _get_loss(self, logits, targets):
        """ Compute loss function based on configs """
        # BxC logits into Bx1 ground truth
        loss = F.cross_entropy(logits, targets)
        return loss
    
    # obtain summary statistics of
    # argmax, max_percentage, entropy for each function
    # expects logits input to be BxC
    def _get_prediction_stats(self, logits):
        prob = F.softmax(logits, dim=1)
        max_percentage, argm = torch.max(prob, dim=1)
        entropy = torch.sum(-prob*torch.log(prob), dim=1)
        return argm, max_percentage, entropy
        
    def _epoch(self, mode, loader):
        losses = []
        gt = []
        pred = []
        # nc = self.num_classes
        # conf = np.zeros((nc, nc))
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        
        for data in tqdm(loader):
            imgs, targets, logits = self._get_network_output(data)
            
            loss = self._get_loss(logits, targets)
            losses.append(loss)
            
            argm, _, _ = self._get_prediction_stats(logits)
            gt.append(targets)
            pred.append(argm)
            # print(' ')
            # print(argm)
            # print(targets)
            
            if mode == 'train':
                # Back-progagate the gradient.
                loss.backward()
                # Update the parameters according to the gradient.
                self.optimizer.step()
                # Zero the parameter gradients in the optimizer
                self.optimizer.zero_grad() 

        loss_avg = torch.mean(torch.stack(losses)).item()
        gt = torch.cat(gt)
        pred = torch.cat(pred)
        acc = sum(gt == pred)/len(gt)
        f1 = f1_score(gt.cpu(), pred.cpu(), average='macro')
        return loss_avg, acc, f1
    
    def train(self, loader_tr, loader_va):
        """Training pipeline."""
            
        best_va_acc = 0.0 # Record the best validation metrics.
        for epoch in range(self.config['num_epochs']):
            loss, acc, f1 = self._epoch('train', loader_tr)
            print(
                "Epoch: %3d, tr L/acc/f1: %.5f/%.3f/%.3f"
                % (epoch, loss, acc, f1)
            )
            # for validation: if torch.no_grad isn't called and loss.backward() also
            # isn't used, the GPU will keep accumulating the gradient which eventually 
            # cause an OOM error.
            # thus the torch.no_grad() before evaluation is mandatory.
            # if there's a more elegant way around this, let me know!
            with torch.no_grad():
                val_loss, val_acc, val_f1 = self._epoch('val', loader_va)
            
            # Save model every epoch.
            self._save(self.checkpts_file)
            if self.config['use_wandb']:
                wandb.log({"tr_loss":loss, "val_loss":val_loss,
                           "tr_acc":acc, "tr_f1":f1,
                           "val_acc":val_acc, "val_f1":val_f1})

            # Early stopping strategy.
            if val_acc > best_va_acc:
                # Save model with the best accuracy on validation set.
                best_va_acc = val_acc
                best_va_f1 = val_f1
                self._save(self.bestmodel_file)
            
            print(
                "Epoch: %3d, val L/acc/f1: %.5f/%.3f/%.3f, top acc/f1: %.3f/%.3f"
                % (epoch, val_loss, val_acc, val_f1, best_va_acc, best_va_f1)
            )
            
            # modify the learning rate
            self.scheduler.step()
    
    @torch.no_grad()
    def test_comprehensive(self, loader, mode="test"):
        """Logs the network outputs in dataloader
        computes per-car preds and outputs result to a DataFrame"""
        print('NOTE: test_comprehensive mode uses batch_size=1 to correctly display metadata')
        # Choose which model to evaluate.
        if mode=="test":
            self._restore(self.bestmodel_file)
        # Switch the model into eval mode.
        self.model.eval()
        
        paths_arr = [[] for _ in range(len(self.car_views))]
        meta_cols = ['make', 'model', 'year', 'car type']
        meta_arr = [[] for _ in range(len(meta_cols))]
        target_arr, pred_arr = [], []
        max_p_arr, entropy_arr = [], []
        
        for data in tqdm(loader):
            imgs = data[0]
            target = data[1]
            paths = data[2]
            misc = data[3]
            
            # collect metadata from data_info
            # we read index zero since it's batchsize 1
            for i in range(len(meta_cols)):
                meta_arr[i].append(misc[i][0])
            for i in range(len(self.car_views)):
                paths_arr[i].append(paths[i][0])
                
            # collect the model prediction info
            _, _, logits = self._get_network_output([imgs, target])
            argm, max_p, ent = self._get_prediction_stats(logits)
            target_arr.append(target.cpu().numpy()[0])
            pred_arr.append(argm.cpu().numpy()[0])
            max_p_arr.append(max_p.cpu().numpy()[0])
            entropy_arr.append(ent.cpu().numpy()[0])
                
        # compile the information into a dictionary
        d = {'label ID':target_arr, 'pred ID': pred_arr, 
             'confidence': max_p_arr, 'entropy': entropy_arr}
        for i in range(len(meta_cols)):
            d[meta_cols[i]] = meta_arr[i]
        for i in range(len(self.car_views)):
            d[self.car_views[i]] = paths_arr[i]
        
        # save the dataframe
        df = pd.DataFrame(data=d)
        test_results_file = os.path.join(self.log_dir, mode+".csv")
        df.to_csv(test_results_file)