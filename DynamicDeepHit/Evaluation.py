import os
from datetime import datetime

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch

from .loss import total_loss, ranking_loss,longitudinal_loss, negative_log_likelihood
from .utils import to_float, compute_brier, compute_cindex

class Evaluation:
     
    def __init__(self,
                 trained_model,
                 test_data,
                 duration_grid_train_np,
                 train_losses,
                 val_losses):

        with open('config.yaml') as yaml_data:
            self.config = yaml.safe_load(yaml_data)

        self.trained_model = trained_model
        self.duration_grid_train_np = duration_grid_train_np
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.X_test_padded, self.Y_train_np, self.D_train_np, self.Y_test_np, self.D_test_np = test_data
        self.duration_grid_test_np = np.unique(self.Y_test_np)
        self.eval_duration_indices = [int(p * len(self.duration_grid_test_np)) for p in [0.25,0.5,0.75]]
        self.events = self.config['events']

    def compute_cifs(self):
        """Compute the Cummulative Incidence Functions for each event"""
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
        
        longit_loss = to_float(longitudinal_loss(self.yhat, self.X_test_padded, kappa=self.config['loss']['kappa']))
        rank_loss   = to_float(ranking_loss(cif, t, e,  1))
        test_nll_loss    = to_float(negative_log_likelihood(self.pmf_test, cif, t, e))
        test_total_loss = to_float(test_nll_loss) + to_float(rank_loss) + to_float(longit_loss)
        return

    # def compute_brier_scores(self):
    #     return compute_brier(self.Y_train_np, self.Y_test_np, self.D_train_np, self.D_test_np,\
    #                             self.events, self.duration_grid_train_np, self.cif_test_np,\
    #                                 self.eval_duration_indices,self.duration_grid_test_np)

    # def compute_cindex_scores(self):
    #     return compute_cindex(self.Y_train_np, self.Y_test_np, self.D_train_np, self.D_test_np,\
    #                         self.events, self.duration_grid_train_np, self.cif_test_np,\
    #                             self.eval_duration_indices,self.duration_grid_test_np)
        
    def plot_total_loss(self, loss_dict, path, constraints = False):
        """Plot total loss"""
        if constraints: model_keys = ['dl1', 'dl2', 'dl3']
        else: model_keys = ['nll', 'rank', 'longit']

        fig, ax = plt.subplots()
        for loss_type_key in loss_dict.keys():
            current_loss_dict = loss_dict[loss_type_key]
            total_loss = np.sum(np.array([current_loss_dict[key] for key in model_keys]), axis = 0)
            plt.plot(total_loss, label=f'{loss_type_key} Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'Total {loss_type_key} Loss')
            ax.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{path}/Total_{loss_type_key}_Loss_plot.png')
        plt.close()


    def plot_losses(self, loss_dict, path, constraints=False):
        """Plot individual loss values with train and validation on the same plot"""
        if constraints: model_keys = ['dl1', 'dl2', 'dl3']
        else: model_keys = ['nll', 'rank', 'longit']
        
        fig, ax = plt.subplots(1, 3, figsize=(12, 7))
        for i, key in enumerate(model_keys):
            for loss_type_key in loss_dict.keys():  # 'train' and 'val'
                current_loss_dict = loss_dict[loss_type_key]
                ax[i].plot(current_loss_dict[key], label=f'{loss_type_key} {key}')
            ax[i].set_xlabel('Epoch')
            ax[i].set_ylabel('Loss')
            ax[i].set_title(f'{key} Loss')
            ax[i].grid()
            ax[i].legend()
        
        plt.tight_layout()
        plt.savefig(f'{path}/Individual_Loss_plot.png')
        plt.close()
    
    
    
    def plot_survival_metrics(self,data):
        fig, ax = plt.subplots()
        plt.plot(concordance_scores['death'], label='CKD')
        plt.plot(base_cindex,'--', label='DDH')
        ax.set_xlabel('horizon')
        ax.set_ylabel('score')
        ax.set_title('Cindex')
        ax.grid()
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{newpath}/Loss_plot_run.png')
        #---------------------------------------------------------#
        # fig, ax = plt.subplots()
        # plt.plot(concordance_scores['death'], label='CKD')
        # plt.plot(base_cindex,'--', label='DDH')
        # ax.set_xlabel('horizon')
        # ax.set_ylabel('score')
        # ax.set_title('Cindex')
        # ax.grid()
        # ax.legend()
        # plt.tight_layout()
        # plt.savefig(f'/home/arnott/git/CKD-JA-MMSc-1/Results/cindex_eGFR/cindex_plots/Run_{run}_{j}.png')
        # # with open(f'/home/arnott/git/CKD-JA-MMSc-1/Results/cindex_eGFR/cindex_cfg/Config_run_{run}_{j}.json', "w") as f:
        # #     cfg['cindex'] = concordance_scores['death']
        # #     json.dump(cfg, f, indent=2)
        # ## TODO - save config yaml
        # #---------------------------------------------------------#
        # fig, ax = plt.subplots()
        # plt.plot(concordance_scores['death'], label='CKD')
        # plt.plot(base_cindex,'--', label='DDH')
        # ax.set_xlabel('horizon')
        # ax.set_ylabel('score')
        # ax.set_title('Cindex')
        # ax.grid()
        # ax.legend()
        # plt.tight_layout()
        # plt.savefig(f'/home/arnott/git/CKD-JA-MMSc-1/Results/cindex_eGFR/cindex_plots/Run_{run}_{j}.png')
        

    def main(self):
        """Main Call"""
        self.compute_cifs()
        # brier_scores = self.compute_brier_scores()
        # cindex_scores = self.compute_cindex_scores()
        brier_scores = 0
        cindex_scores = 0
        if self.config['results']['save_results']:
            self.store_results(self.train_losses, self.val_losses)