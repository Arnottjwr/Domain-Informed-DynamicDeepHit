import json
import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from ddh.ddh_torch import DynamicDeepHitTorch
from ddh.losses_CRexp2 import total_loss, ranking_loss,longitudinal_loss, negative_log_likelihood

from utils import load_data, compute_brier_competing, discretize, _get_padded_features, zip_features, init_scheduler


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

        Y_train_discrete_np, self.duration_grid_train_np = discretize(self.Y_train_list, {self.config["num_durations"]})
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

        best_val_loss = float('inf')
        best_val = float('inf')
        best_state = None

        train_losses = {'l1':[],'l2':[],'l3':[], "dl1": [],"dl2": [], "dl3": []}
        val_losses = {'l1':[], 'l2':[],'l3':[], "dl1": [],"dl2": [], "dl3": []}

        for epoch in range(training_config['num_epochs']):
            # training_params['gamma'] = warmup_gammas(epoch,(cfg['gamma_1'],cfg['gamma_2'],cfg['gamma_3'], 0),warmup_epochs=25) ## TODO - implement this
            # ---- TRAIN ----
            self.dynamic_deephit_model.train()
            for Xb, Yb, Db in train_loader:
                Xb = Xb.to(self.device)
                # Yb, Db = Yb.to(device), Db.to(device)

                # Optional: schedule gamma via training_params['gamma'] here if you want warm-up
                # training_params['gamma'] = scheduled_gammas(epoch)

                train_loss_scalar = total_loss(self.dynamic_deephit_model, Xb, Yb, Db, loss_config,
                                                                for_selection=False,        # <— include domain in training loss
                                                                include_rank_in_selection=True,
                                                                compute_parts=True,
                                                                detach_domain_from_trunk=False  # try True if domain hurts representation
                                                            )

                optimiser.zero_grad(set_to_none=True)
                train_loss_scalar.backward()
                torch.nn.utils.clip_grad_norm_(self.dynamic_deephit_model.parameters(), max_norm=grad_clip)
                optimiser.step()

            # ---- EVAL ----
            self.dynamic_deephit_model.eval()
            with torch.no_grad():
                # --- Train loss ---
                tr_log, tr_n = {"l1":0,"l2":0,"l3":0,"dl1": 0,"dl2": 0, "dl3": 0}, 0
                for Xb, Yb, Db in train_loader:
                    Xb = Xb.to(self.device)
                    parts = total_loss(self.dynamic_deephit_model, Xb, Yb, Db, loss_config,
                                                for_selection=False, compute_parts=True)
                    current_batch_size = Xb.size(0)
            
                    for k in tr_log: 
                        tr_log[k] += parts[k] * current_batch_size
                    tr_n += current_batch_size

                for k in tr_log: 
                    tr_log[k] /= tr_n
                    train_losses[k].append(tr_log[k])

                




                # ---Validation---
                va_log, va_sel,va_dom, va_model, va_n = {"model":0,"domain":0,"longit":0,"rank":0,"nll":0,"total":0}, 0.0, 0.0, 0.0,0
                for Xb, Yb, Db in val_loader:
                    Xb = Xb.to(self.device)
                    # Compute once; we'll pick out the parts we want
                    sel_loss, parts = total_loss(self.dynamic_deephit_model, Xb, Yb, Db, loss_config,
                                        for_selection=True, compute_parts=True)
                    _, dom_parts = total_loss(self.dynamic_deephit_model, Xb, Yb, Db, loss_config,
                                                for_selection=False, compute_parts=True)

                    bs = Xb.size(0)
                    va_sel += float(sel_loss) * bs
                    va_dom += dom_parts['domain']*bs
                    for k in va_log:
                        va_log[k] += float(parts[k]) * bs
                    va_n += bs

                va_sel/=va_n
                va_dom/=va_n
                for k in va_log:
                    va_log[k] /= va_n

                # Use ONLY NLL for early stopping / LR scheduling
                val_nll = va_log["nll"]
                val_nll_epoch_losses.append(val_nll)
                val_epoch_losses.append(va_sel)
                val_domain_losses.append(va_dom)
                val_model_losses.append(va_log['model'])
            scheduler.step(val_nll)
            

            print(f"[{epoch+1}] train_total={tr_log['total']:.4f} "
                f"model_loss={tr_log['model']:.4f}  domain_loss={tr_log['domain']:.4f} "
                f"(nll={tr_log['nll']:.4f}, rank={tr_log['rank']:.4f}, longit={tr_log['longit']:.4f})")

            # checkpoint on validation selection metric ONLY
            if val_nll < best_val:
                best_val = val_nll
                best_state = {k: v.detach().cpu().clone() for k, v in self.dynamic_deephit_model.state_dict().items()}


        if best_state is not None:
            self.dynamic_deephit_model.load_state_dict(best_state)




    def run_evaluation_metrics(self):
        pass

    def store_results(self):
        pass




    def main(self):
        """Main call function"""
        print(self.config)
