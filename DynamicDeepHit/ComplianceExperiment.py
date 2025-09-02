import os
import json
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lifelines
from torch.utils.data import DataLoader
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from ddh.ddh_torch import DynamicDeepHitTorch
from ddh.losses_CRexp3 import total_loss, ranking_loss,longitudinal_loss, negative_log_likelihood
from utils import discretize, compute_brier_competing, load_data, _get_padded_features
from copy import deepcopy
import torch

def to_float(x):
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    return float(x)

cuda = 'cuda'
run = 3
dataset = '/home/arnott/git/CKD-JA-MMSc-1/Data/real_eGFR_mimic.csv'
features_before_preprocessing = ['gender','age_calculated','Creatinine','Hemoglobin','eGFR','UPCR','ACUR']
events = ['death']

df = pd.read_csv(dataset)

df[df.isna()] = -1 ##NOTE - Mask missing features with -1
df['gender'] = np.where(df['gender']=='F',1,0) # female = 1, male = 0
df = df[df['subject_id'].map(df['subject_id'].value_counts()) > 1] # remove tabular data
df = df.sort_values(['subject_id', 'year'], ascending=[True, True]).reset_index(drop=True)


features = df[features_before_preprocessing].to_numpy().astype('float32')    
observed_times = (df['years'] - df['year']).to_numpy().astype('float32') # Calculate Time to Events
event_indicators = (df['event'] == 'death').to_numpy().astype('int32') # 0 == survived (censored), 1 == death, 2== dialysis
event_indicators[df['event'] == 'dialysis'] = 2
X = []
Y = []
D = []
for id in sorted(list(set(df['subject_id']))):
    mask = (df['subject_id'] == id)
    X.append(features[mask])
    Y.append(observed_times[mask])
    D.append(event_indicators[mask])

X_full_train_raw_list, X_test_list, Y_full_train_list, Y_test_list, D_full_train_list, D_test_list = train_test_split(X, Y, D, test_size=.3, random_state=0)

# split the "full training set" into the actual training set and a validation set (using a 80/20 split)
X_train_list, X_val_list, Y_train_list, Y_val_list, D_train_list, D_val_list = train_test_split(X_full_train_raw_list, Y_full_train_list, D_full_train_list, test_size=.2, random_state=0)

num_durations = 256  # set this to 0 to use all unique durations in which any critical event happens
Y_train_discrete_np, duration_grid_train_np = discretize(Y_train_list, num_durations)
Y_val_discrete_np, _ = discretize(Y_val_list, len(duration_grid_train_np) - 1, duration_grid_train_np)
output_num_durations = len(duration_grid_train_np)

device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
X_train_padded = torch.from_numpy(_get_padded_features(X_train_list)).type(torch.float32).to(device)

last_Y_train = torch.tensor([_[-1] for _ in Y_train_discrete_np]).type(torch.float32).to(device)
D_train = torch.tensor([_[-1] for _ in D_train_list]).type(torch.float32).to(device)

train_data = list(zip(X_train_padded, last_Y_train, D_train))

X_val_padded = torch.from_numpy(_get_padded_features(X_val_list)).type(torch.float32).to(device)
last_Y_val = torch.tensor([_[-1] for _ in Y_val_discrete_np]).type(torch.float32).to(device)
D_val = torch.tensor([_[-1] for _ in D_val_list]).type(torch.float32).to(device)
val_data = list(zip(X_val_padded, last_Y_val, D_val))

# RNN parameters
num_hidden = 96  # RNN hidden layer number of nodes (i.e., dimension of RNN hidden layer); `width` [8,16,32,64]
num_rnn_layers = 2 # number of lstm layers `depth` [2,4,8]
rnn_type = 'LSTM'  # 'RNN', 'LSTM', and 'GRU'
dropout = 0.18423243369011189

# number of nodes for different hidden layers in longitudinal 
layers_for_predicting_next_time_step = [64,64]

# number of nodes for different hidden layers in f_{attention}
layers_for_attention = [64,64]

# number of nodes for different hidden layers in each cause specific network 
layers_for_each_deephit_event = [64,64]
torch.manual_seed(0)

num_input_features = X_train_padded.size(2)
dynamic_deephit_model = \
    DynamicDeepHitTorch(input_dim = num_input_features,
                        output_dim = output_num_durations,
                        layers_rnn = num_rnn_layers,
                        hidden_rnn = num_hidden,
                        long_param={'layers': layers_for_predicting_next_time_step,
                                    'dropout': dropout},
                        att_param={'layers': layers_for_attention,
                                   'dropout': dropout},
                        cs_param={'layers': layers_for_each_deephit_event,
                                  'dropout': dropout},
                        typ=rnn_type,
                        risks=len(events)).to(device)
dynamic_deephit_loss = total_loss

use_constraints = True
training_params =  {"alpha": 0.15598768146289432, # rank loss
                    "beta":  1, # NLL loss 
                    "delta": 0.08538145469419971,   # longit loss
                    "sigma": 0.35434828698561155,
                    # "gamma": [0,0,0, 0],
                    "gamma": [1,0,0, 0],
                    "bound_dict": {3: [0,17.5],  # Cr 2: [0,17.0]    0.336,0.026,0.181,        
                                   4: [0,128.3014697269145],
                                   5: [0,17.6],
                                   6: [0,12767]
                                   },
                    "use_constraints":use_constraints,
                    }

num_epochs = 5000
batch_size = 64 # 128,256,512,1028
learning_rate = 0.00325807701220510668
grad_clip = 0.9034919529397347
threshold = 0
eps = 1e-3
patience = 10
best_avg = float('inf')



train_loader = DataLoader(train_data, batch_size, shuffle=True)  # shuffling for minibatch gradient descent
val_loader = DataLoader(val_data, batch_size, shuffle=False)  # there is no need to shuffle the validation data
optimizer = torch.optim.AdamW(dynamic_deephit_model.parameters(), lr=learning_rate, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',      # because we want to minimize validation loss
    factor=0.5,      # multiply lr by 0.5 each time
    patience=8,      # wait 5 epochs of no improvement before reducing
    threshold=5e-4,
    threshold_mode='rel',
    min_lr= 1e-5
)

train_epoch_losses = []
val_sel_epoch_losses = []
val_dom_epoch_losses = []
val_nll_epoch_losses = []
train_nll_epoch_losses = []

dl1_pct_violate = []
dl2_pct_violate = []
dl3_pct_violate = []

dl1_Lc_list = []
dl1_Lv_list = []



best_val_loss = float('inf')
best_params = None
best_epoch_index = None


best_val = float('inf')
best_state = None

for epoch in range(num_epochs):
    # ---- TRAIN ----
    dynamic_deephit_model.train()
    for Xb, Yb, Db in train_loader:
        Xb = Xb.to(device)
        # Move Yb/Db to device if your loss expects it on GPU
        # Yb, Db = Yb.to(device), Db.to(device)

        # Optional: schedule gamma via training_params['gamma'] here if you want warm-up
        # training_params['gamma'] = scheduled_gammas(epoch)

        (train_loss_scalar, train_parts) = total_loss(dynamic_deephit_model, Xb, Yb, Db, training_params,
                                                        for_selection=False,        # <— include domain in training loss
                                                        include_rank_in_selection=True,
                                                        compute_parts=True,
                                                        detach_domain_from_trunk=False  # try True if domain hurts representation
                                                    )

        optimizer.zero_grad(set_to_none=True)
        train_loss_scalar.backward()
        torch.nn.utils.clip_grad_norm_(dynamic_deephit_model.parameters(), max_norm=grad_clip)
        optimizer.step()

    # ---- EVAL ----
    dynamic_deephit_model.eval()
    with torch.no_grad():
        # aggregate train logs (optional)
        sum_Lc, sum_Lv = 0.0, 0.0
        N_Lc, N_Lv = 0,0 
        dl1_violations, dl2_violations, dl3_violations = 0.0, 0.0, 0.0
        dl1_total_points, dl2_total_points, dl3_total_points = 0.0,0.0,0.0
        tr_log, tr_n = {"model":0,"domain":0,"longit":0,"rank":0,"nll":0,"total":0}, 0

        for Xb, Yb, Db in train_loader:
            Xb = Xb.to(device)
            loss_val, parts = total_loss(dynamic_deephit_model, Xb, Yb, Db, training_params,
                                         for_selection=False, compute_parts=True)


            sum_Lc += float(parts['dl1_Lc_sum'])
            sum_Lv += float(parts['dl1_Lv_sum'])
            N_Lc += int(parts['dl1_Lc_count'])
            N_Lv += int(parts['dl1_Lv_count'])

            current_batch_size = Xb.size(0)
            for k in tr_log: 
                tr_log[k] += parts[k] * current_batch_size
            tr_n += current_batch_size
    
        if N_Lc != 0:
            dl1_Lc = sum_Lc/N_Lc 
            dl1_Lc_list.append(dl1_Lc)

        if N_Lv != 0:
            dl1_Lv = sum_Lv/N_Lv
            dl1_Lv_list.append(dl1_Lv)

        # dl1_Lv = (sum_Lv / max(N_Lv,1)) if N_Lv> 0 else float('nan')


        for k in tr_log:
            tr_log[k] /= tr_n
        train_epoch_losses.append(tr_log['total'])
        train_nll_epoch_losses.append(tr_log['nll'])

        va_log, va_sel,va_dom, va_n = {"domain":0,"longit":0,"rank":0,"nll":0,"total":0}, 0.0, 0.0, 0
        for Xb, Yb, Db in val_loader:
            Xb = Xb.to(device)
            # Compute once; we'll pick out the parts we want
            sel_loss, parts = total_loss(dynamic_deephit_model, Xb, Yb, Db, training_params,
                                for_selection=True, compute_parts=True)
   

            bs = Xb.size(0)
            va_sel += float(sel_loss) * bs
            va_dom += parts['domain']*bs
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
        val_sel_epoch_losses.append(va_sel)
        val_dom_epoch_losses.append(va_dom)
    scheduler.step(val_nll)
    

    print(f"[{epoch+1}] model_loss={tr_log['model']:.4f} "
          f"domain_loss={tr_log['domain']:.4f}  val_domain={va_log['domain']:.4f} "
          f"(nll={tr_log['nll']:.4f}, rank={tr_log['rank']:.4f}, longit={tr_log['longit']:.4f})")
    
    # checkpoint on validation selection metric ONLY
    if val_nll < best_val:
        best_val = val_nll
        best_state = {k: v.detach().cpu().clone() for k, v in dynamic_deephit_model.state_dict().items()}


# restore best
if best_state is not None:
    dynamic_deephit_model.load_state_dict(best_state)


plt.plot(dl1_Lc_list, label='Compliant')
plt.plot(dl1_Lv_list,label='Non-compliant')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0,15000)
plt.title('Compliant and Violant Loss Values')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(f'/home/arnott/git/CKD-JA-MMSc-1/Results/ComplianceExperiment/Run_{run}_Constraints_{use_constraints}')


# # Parameters
params = {
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    "dropout": dropout,
    "threshold":threshold,
    "grad_clip":grad_clip,
    "layers_for_attention": layers_for_attention,
    "layers_for_each_deephit_event": layers_for_each_deephit_event,
    "layers_for_predicting_next_time_step": layers_for_predicting_next_time_step,
    "num_hidden": num_hidden,
    "num_rnn_layers": num_rnn_layers,
    "rnn_type": rnn_type,
    "num_durations": num_durations,
    "num_epochs": num_epochs,
}

data = {
    "dataset": dataset,
    "params":params,
    "training_params": training_params,
    "compliant_noncompliant_points":(dl1_Lc_list,dl1_Lv_list),
    "loss":{
        "epochs":num_epochs,
        "train_losses":train_epoch_losses,
        "val_nll_epoch_losses":val_nll_epoch_losses,
        },
    }
output_dir = "/home/arnott/git/CKD-JA-MMSc-1/Results/ComplianceExperiment"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"Run_{run}_constraints_{use_constraints}.json")

# Save to JSON with keys
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)
print(f'Data saved under {output_path}')