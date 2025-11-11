import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from .ddh.ddh_torch import DynamicDeepHitTorch
from .ddh.loss import total_loss

class Trainer:
    def __init__(self, 
                 train_data,
                 X_train_padded,
                 val_data,
                 output_num_durations,
                 device):

        np.random.seed(self.config['misc']['seed'])
        with open('config.yaml') as yaml_data:
            self.config = yaml.safe_load(yaml_data)
        self.device = device
        model_config = self.config['model']
        
        self.train_data = train_data
        self.val_data = val_data

        dropout = model_config['dropout']
        num_input_features = X_train_padded.size(2)
        self.dynamic_deephit_model = DynamicDeepHitTorch(
                                            input_dim = num_input_features,
                                            output_dim = output_num_durations,
                                            layers_rnn = model_config['num_rnn_layers'],
                                            hidden_rnn = model_config['num_hidden'],
                                            long_param={'layers': model_config['layers_for_predicting_next_time_step'],
                                                'dropout': dropout},
                                            att_param={'layers': model_config['layers_for_attention'],
                                                        'dropout': dropout},
                                            cs_param={'layers': model_config['layers_for_each_deephit_event'],
                                                        'dropout': dropout},
                                            typ=model_config['rnn_type'],
                                            risks=len(self.config['events'])).to(self.device)
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
        scheduler = self.init_scheduler(optimiser, self.config['lr_scheduler'])

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

        return self.dynamic_deephit_model

    def init_scheduler(self, optimiser, config):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser,
                mode = config['mode'],
                factor=config['factor'],
                patience=config['patience'],
                threshold=config['threshold'],
                threshold_mode=config['threshold_mode'],
                min_lr= config['min_lr']
            )


    def main(self):
        """Main call function"""
        pass
