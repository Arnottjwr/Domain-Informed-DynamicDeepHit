import json
import yaml
import os
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from ddh.ddh_torch import DynamicDeepHitTorch
from ddh.loss import total_loss, ranking_loss,longitudinal_loss, negative_log_likelihood

from utils import discretize, _get_padded_features, zip_features, init_scheduler, to_float, compute_brier, compute_cindex


class DomainInformedModel:

    def __init__(self):
        with open('config.yaml') as yaml_data:
            self.config = yaml.safe_load(yaml_data)

        np.random.seed(self.config['misc']['seed'])
        self.events = self.config['events']
        self.device = torch.device(self.config['misc']['cuda'] if torch.cuda.is_available() else 'cpu')


    """
    - initialise model params 
    - run training loop 
    - run evaluation metrics
    - store config and evaluation metrics
    """
    
    def load_data(self):
        """Load the dataset and preproccess it"""
        local_config = self.config['load_data']
        df = pd.read_csv(local_config['path'])
        
        features = df[local_config['features_before_preprocessing']].to_numpy().astype('float32')    
        observed_times = (df['years'] - df['year']).to_numpy().astype('float32') # Calculate Time to Events
        event_indicators = df['event'].map({event:i+1 for i, event in enumerate(self.events)}).fillna(0).astype('int32').to_numpy() ##TODO - Test this
        X = []
        Y = []
        D = []
        for id in sorted(list(set(df['subject_id']))):
            mask = (df['subject_id'] == id)
            X.append(features[mask])
            Y.append(observed_times[mask])
            D.append(event_indicators[mask])

        X_full_train_raw_list, self.X_test_list, Y_full_train_list, self.Y_test_list, D_full_train_list, self.D_test_list = train_test_split(X, Y, D, test_size=.3, random_state=0)
        X_train_list, X_val_list, self.Y_train_list, Y_val_list, self.D_train_list, D_val_list = train_test_split(X_full_train_raw_list, Y_full_train_list, D_full_train_list, test_size=.2, random_state=0)

        Y_train_discrete_np, self.duration_grid_train_np = discretize(self.Y_train_list, self.config["num_durations"])
        Y_val_discrete_np, _ = discretize(Y_val_list, len(self.duration_grid_train_np) - 1, self.duration_grid_train_np)
        self.output_num_durations = len(self.duration_grid_train_np)

        self.train_data, self.X_train_padded = zip_features(X_train_list, Y_train_discrete_np, self.D_train_list,device=self.device)
        self.val_data, _ = zip_features(X_val_list, Y_val_discrete_np, D_val_list,device=self.device)


    def init_model(self):
        model_config = self.config['model']
        dropout = model_config['dropout']
        num_input_features = self.X_train_padded.size(2)
        self.dynamic_deephit_model = DynamicDeepHitTorch(
                                            input_dim = num_input_features,
                                            output_dim = self.output_num_durations,
                                            layers_rnn = model_config['num_rnn_layers'],
                                            hidden_rnn = model_config['num_hidden'],
                                            long_param={'layers': model_config['layers_for_predicting_next_time_step'],
                                                'dropout': dropout},
                                            att_param={'layers': model_config['layers_for_attention'],
                                                        'dropout': dropout},
                                            cs_param={'layers': model_config['layers_for_each_deephit_event'],
                                                        'dropout': dropout},
                                            typ=model_config['rnn_type'],
                                            risks=len(self.events)).to(self.device)
        self.dynamic_deephit_loss = total_loss

    def train_and_validate(self):
        """Train the model"""
        training_config = self.config['training']
        batch_size = training_config['batch_size']
        grad_clip = training_config['grad_clip']

        loss_config = self.config['loss']
        self.use_constraints = loss_config['use_constraints']


        train_loader = DataLoader(self.train_data, batch_size, shuffle=True)  # shuffling for minibatch gradient descent
        val_loader = DataLoader(self.val_data, batch_size, shuffle=False)  # there is no need to shuffle the validation data
        optimiser = torch.optim.AdamW(self.dynamic_deephit_model.parameters(), lr=training_config['learning_rate'], weight_decay=1e-3)
        scheduler = init_scheduler(optimiser,self.config['lr_scheduler'])

        best_val = float('inf')
        best_state = None

        self.train_losses = {'nll':[],'rank':[],'longit':[], "dl1": [],"dl2": [], "dl3": []}
        self.val_losses = {'nll':[], 'rank':[],'longit':[], "dl1": [],"dl2": [], "dl3": []}

        for epoch in range(training_config['num_epochs']):
            # training_params['gamma'] = warmup_gammas(epoch,(cfg['gamma_1'],cfg['gamma_2'],cfg['gamma_3'], 0),warmup_epochs=25) ## TODO - implement this
            # ---- TRAIN ----
            self.dynamic_deephit_model.train()
            for Xb, Yb, Db in train_loader:
                Xb,Yb, Db = Xb.to(self.device), Yb.to(self.device), Db.to(self.device)

                # Optional: schedule gamma via training_params['gamma'] here if you want warm-up
                # training_params['gamma'] = scheduled_gammas(epoch)

                train_loss_scalar = total_loss(self.dynamic_deephit_model, Xb, Yb, Db, loss_config)
                optimiser.zero_grad(set_to_none=True)
                train_loss_scalar.backward()
                torch.nn.utils.clip_grad_norm_(self.dynamic_deephit_model.parameters(), max_norm=grad_clip)
                optimiser.step()

            # ---- EVAL ----
            self.dynamic_deephit_model.eval()
            with torch.no_grad():
                # --- Train loss ---
                tr_log, tr_n = {"nll":0,"rank":0,"longit":0,"dl1": 0,"dl2": 0, "dl3": 0}, 0
                for Xb, Yb, Db in train_loader:
                    Xb = Xb.to(self.device)
                    parts = total_loss(self.dynamic_deephit_model, Xb, Yb, Db, loss_config, eval_flag=True)
                    current_batch_size = Xb.size(0)
            
                    for k in tr_log: 
                        tr_log[k] += parts[k] * current_batch_size
                    tr_n += current_batch_size

                for k in tr_log: 
                    tr_log[k] /= tr_n
                    self.train_losses[k].append(tr_log[k])


                # ---Validation---
                va_log, va_n = {"nll":0,"rank":0,"longit":0, "dl1": 0,"dl2": 0, "dl3": 0}, 0
                for Xb, Yb, Db in val_loader:
                    Xb = Xb.to(self.device)
                    # Compute once; we'll pick out the parts we want
                    va_parts = total_loss(self.dynamic_deephit_model, Xb, Yb, Db, loss_config, eval_flag=True)
                    current_batch_size = Xb.size(0)

                    for k in va_log: 
                        va_log[k] += va_parts[k] * current_batch_size
                    va_n += current_batch_size

                for k in va_log: 
                    va_log[k] /= va_n
                    self.val_losses[k].append(va_log[k])

        
                # Use ONLY NLL for early stopping / LR scheduling
                val_nll = va_log["nll"]

            scheduler.step(val_nll)
            
            tr_log_model = tr_log['nll'] + tr_log['rank'] + tr_log['longit']
            tr_log_domain = tr_log['dl1'] + tr_log['dl2'] + tr_log['dl3']
            print(f"[{epoch+1}] train_total={tr_log_model + tr_log_domain:.4f} "
                f"model_loss={tr_log_model:.4f}  domain_loss={tr_log_domain:.4f} "
                f"(nll={tr_log['nll']:.4f}, rank={tr_log['rank']:.4f}, longit={tr_log['longit']:.4f})")

            # checkpoint on validation selection metric ONLY
            if val_nll < best_val:
                best_val = val_nll
                best_state = {k: v.detach().cpu().clone() for k, v in self.dynamic_deephit_model.state_dict().items()}


        if best_state is not None:
            self.dynamic_deephit_model.load_state_dict(best_state)




    def run_evaluation_metrics(self):
        X_test_padded = torch.from_numpy(_get_padded_features(self.X_test_list)).type(torch.float32).to(self.device)

        # Evaluate on Test Data
        with torch.no_grad():
            yhat, pmf_test = self.dynamic_deephit_model(X_test_padded)

        cifs = []
        for event_idx_minus_one in range(len(pmf_test)):
            cifs.append(pmf_test[event_idx_minus_one].cumsum(1))
        cif_test_np = np.array([cifs[event_idx_minus_one].cpu().numpy().T for event_idx_minus_one in range(len(self.events))])
        cif_test_np = cif_test_np.tolist()
        cif_test_np = np.array(cif_test_np)

        Y_train_np = np.array([_[-1] for _ in self.Y_train_list])
        D_train_np = np.array([_[-1] for _ in self.D_train_list])
        Y_test_np = np.array([_[-1] for _ in self.Y_test_list])
        D_test_np = np.array([_[-1] for _ in self.D_test_list])

        # Loss Evaluation

        t = torch.tensor(Y_test_np).long()
        e = torch.tensor(D_test_np).int()
        cif = [torch.cumsum(ok, dim=1) for ok in pmf_test]
        longit_loss = longitudinal_loss(yhat, X_test_padded, kappa=self.config['loss']['kappa'])
        rank_loss   = ranking_loss(cif, t, e,  1)
        test_nll_loss    = negative_log_likelihood(pmf_test, cif, t, e)
        test_total_loss = to_float(test_nll_loss) + to_float(rank_loss) + to_float(longit_loss)
        test_nll_loss = to_float(test_nll_loss)


        duration_grid_test_np = np.unique(Y_test_np)
        eval_duration_indices = [int(p * len(duration_grid_test_np)) for p in [0.25,0.5,0.75]]


        self.brier_scores = compute_brier(Y_train_np, Y_test_np, D_train_np, D_test_np,\
                                self.events, self.duration_grid_train_np, cif_test_np,\
                                    eval_duration_indices,duration_grid_test_np)
        
        self.cindex_scores = compute_cindex(Y_train_np, Y_test_np, D_train_np, D_test_np,\
                                self.events, self.duration_grid_train_np, cif_test_np,\
                                    eval_duration_indices,duration_grid_test_np)

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






    def main(self):
        """Main call function"""
        self.load_data()
        self.init_model()
        self.train_and_validate()
        self.run_evaluation_metrics()
        self.store_results()
