import yaml
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch

from ddh.loss import total_loss, ranking_loss,longitudinal_loss, negative_log_likelihood
from utils import to_float, compute_brier, compute_cindex

class Evaluation:
     
    def __init__(self,
                 trained_model,
                 test_data,
                 duration_grid_train_np):

        with open('config.yaml') as yaml_data:
            self.config = yaml.safe_load(yaml_data)

        self.trained_model = trained_model
        self.duration_grid_train_np = duration_grid_train_np
        self.X_test_padded, self.Y_train_np, self.D_train_np, self.Y_test_np, self.D_test_np = test_data


        self.duration_grid_test_np = np.unique(self.Y_test_np)
        self.eval_duration_indices = [int(p * len(self.duration_grid_test_np)) for p in [0.25,0.5,0.75]]
        self.events =self.config['events']

    def compute_cifs(self):
        with torch.no_grad():
            self.yhat, self.pmf_test = self.trained_model(self.X_test_padded)

        cifs = []
        for event_idx_minus_one in range(len(self.pmf_test)):
            cifs.append(self.pmf_test[event_idx_minus_one].cumsum(1))
        cif_test_np = np.array([cifs[event_idx_minus_one].cpu().numpy().T for event_idx_minus_one in range(len(self.events))])
        cif_test_np = cif_test_np.tolist()
        self.cif_test_np = np.array(cif_test_np)

    
    def compute_loss_functions(self):
        """Loss Evaluation"""
        t = torch.tensor(self.Y_test_np).long()
        e = torch.tensor(self.D_test_np).int()
        cif = [torch.cumsum(ok, dim=1) for ok in self.pmf_test]
        longit_loss = longitudinal_loss(self.yhat, self.X_test_padded, kappa=self.config['loss']['kappa'])
        rank_loss   = ranking_loss(cif, t, e,  1)
        test_nll_loss    = negative_log_likelihood(self.pmf_test, cif, t, e)
        test_total_loss = to_float(test_nll_loss) + to_float(rank_loss) + to_float(longit_loss)
        self.test_nll_loss = to_float(test_nll_loss)


    def compute_brier_scores(self):
        return compute_brier(self.Y_train_np, self.Y_test_np, self.D_train_np, self.D_test_np,\
                                self.events, self.duration_grid_train_np, self.cif_test_np,\
                                    self.eval_duration_indices,self.duration_grid_test_np)

    def compute_cindex_score(self):
        return compute_cindex(self.Y_train_np, self.Y_test_np, self.D_train_np, self.D_test_np,\
                            self.events, self.duration_grid_train_np, self.cif_test_np,\
                                self.eval_duration_indices,self.duration_grid_test_np)

    def store_results(self):
        date_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        newpath = rf'Data\\ExperiementalData\\{date_time}' 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if self.config['results']['plots']:
            _, ax = plt.subplots()
            plt.plot(model_losses, label='Model Loss')
            plt.plot(domain_losses,'--', label='Domain Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Model and Domain Loss')
            ax.grid()
            ax.legend()
            plt.tight_layout()
            plt.savefig(f'{newpath}/Loss_plot_run.png')
        ## TODO - plot brier scores
        ## TODO - plot cindex scores
        ## TODO - save config yaml