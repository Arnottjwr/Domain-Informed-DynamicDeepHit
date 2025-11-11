
import os
import json
import math
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from ddh.ddh_torch import DynamicDeepHitTorch
from ddh.losses_CRexp2 import total_loss, ranking_loss,longitudinal_loss, negative_log_likelihood
from copy import deepcopy
import torch
import lifelines
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
import itertools, random
from utils import load_data, compute_brier_competing, discretize, _get_padded_features
from torch import nn


def sample_config(search_space: dict) -> dict:
    """
    search_space example:
      {
        "lr": {"type": "loguniform", "low": 1e-4, "high": 3e-2},
        "hidden": {"type": "choice", "vals": [64, 128, 256]},
        "dropout": {"type": "uniform", "low": 0.0, "high": 0.5},
        "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-2},
      }
    """
    cfg = {}
    for k, spec in search_space.items():
        if spec["type"] == "choice":
            cfg[k] = random.choice(spec["vals"])
        elif spec["type"] == "uniform":
            a, b = spec["low"], spec["high"]
            cfg[k] = a + (b - a) * random.random()
        elif spec["type"] == "loguniform":
            lo, hi = math.log(spec["low"]), math.log(spec["high"])
            cfg[k] = math.exp(lo + (hi - lo) * random.random())
        else:
            raise ValueError(f"Unknown type for {k}")
    return cfg

def rolling_average(arr, window):
    return float(np.mean(arr[-window:])) if len(arr) >= window else float('inf')

def sample_patients(df,n=1000, id_col='subject_id',random_state=None):
    ids = df[id_col]
    unique_ids = ids.drop_duplicates()
    sampled_ids = unique_ids.sample(n=n,random_state =random_state)
    return df[df[id_col].isin(sampled_ids)]

def warmup_gammas(epoch, target_gammas, warmup_epochs = 25):
    factor = min(1, (epoch+1)/warmup_epochs)
    return tuple(g * factor for g in target_gammas)

def to_float(x):
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    return float(x)
print('chnages')
cuda = 'cuda:2'
dataset = '/home/arnott/git/CKD-JA-MMSc-1/Data/mimic_extra_features.csv'
features_before_preprocessing = ['gender','age_calculated','Creatinine','Hemoglobin','eGFR','UPCR','ACUR','sbp','dbp','mbp']
events = ['death','dialysis']
np.random.seed(1112)

df = pd.read_csv(dataset)
df = df.sort_values(['subject_id','year'],ascending = [True,True]).reset_index(drop=True)
df[df.isna()] = -1 ##NOTE - Mask missing features with -1
df['gender'] = np.where(df['gender']=='F',1,0) # female = 1, male = 0
df = df[df['subject_id'].map(df['subject_id'].value_counts()) > 1] # remove tabular data
df = sample_patients(df,n=775,random_state=1111)

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
X_train_list, X_val_list, Y_train_list, Y_val_list, D_train_list, D_val_list = train_test_split(X_full_train_raw_list, Y_full_train_list, D_full_train_list, test_size=.2, random_state=0)
 
num_durations = 256  # set this to 0 to use all unique durations in which any critical event happens
Y_train_discrete_np, duration_grid_train_np = discretize(Y_train_list, num_durations)
Y_val_discrete_np, _ = discretize(Y_val_list, len(duration_grid_train_np) - 1, duration_grid_train_np)
output_num_durations = len(duration_grid_train_np)
device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
X_train_padded = torch.from_numpy(_get_padded_features(X_train_list)).type(torch.float32).to(device)
# X_train_padded.size()  # shape = (number of data points, max number of time steps, number of features per time step)
last_Y_train = torch.tensor([_[-1] for _ in Y_train_discrete_np]).type(torch.float32).to(device)
D_train = torch.tensor([_[-1] for _ in D_train_list]).type(torch.float32).to(device)
# each of these will be a 1D table with number of entries given by the number of data points
train_data = list(zip(X_train_padded, last_Y_train, D_train))
X_val_padded = torch.from_numpy(_get_padded_features(X_val_list)).type(torch.float32).to(device)
last_Y_val = torch.tensor([_[-1] for _ in Y_val_discrete_np]).type(torch.float32).to(device)
D_val = torch.tensor([_[-1] for _ in D_val_list]).type(torch.float32).to(device)
val_data = list(zip(X_val_padded, last_Y_val, D_val))
# --------------------------------------------------------------------------------------------------------------------------------------------

run = 2
use_constraints = True
best_avg = float('inf')



# RNN parameters
num_hidden = 96  # RNN hidden layer number of nodes (i.e., dimension of RNN hidden layer); `width` [8,16,32,64]
num_rnn_layers = 2 # number of lstm layers `depth` [2,4,8]
rnn_type = 'LSTM'  # 'RNN', 'LSTM', and 'GRU'
dropout = 0.25

# number of nodes for different hidden layers in longitudinal 
layers_for_predicting_next_time_step = [64,64] 

# number of nodes for different hidden layers in f_{attention}
layers_for_attention = [64,64] 

# number of nodes for different hidden layers in each cause specific network 
layers_for_each_deephit_event = [64,64]
torch.manual_seed(0)


training_params =  {"alpha": 0.15598768146289432, # rank loss
                "beta":  1, # NLL loss 
                "delta": 0.08538145469419971,   # longit loss
                "sigma": 0.35434828698561155,
                # "gamma": [0.241,0.291,0.266, 0],
                "gamma":[None,None,None,0],
                "bound_dict": {3: [0,17.5],  # Cr 2: [0,17.0]    0.336,0.026,0.181,        
                                4: [0,128.3014697269145],
                                5: [0,17.6],
                                6: [0,12767],
                                },
                "use_constraints":use_constraints,
                }

num_epochs = 5000
batch_size = 64 # 128,256,512,1028
learning_rate = 0.00345807701220510668
grad_clip = 0.9034919529397347

base_brier_death = [np.float64(0.2124447197612494), np.float64(0.2493920893728955), np.float64(0.22674590046303772), np.float64(0.16505498135499228)]
base_brier_dialysis = [np.float64(0.07123707871568459), np.float64(0.07123702680258785), np.float64(0.07123696004647606), np.float64(0.07123660643444835)]
base_cindex_death = [np.float64(0.8324586192975374), np.float64(0.7820109325824088), np.float64(0.7772664519098021), np.float64(0.716867469879518)]
base_cindex_dialysis = [np.float64(0.9784753363228698), np.float64(0.9784753363228698), np.float64(0.9784753363228698), np.float64(0.9784753363228698)]

base_config = {
    "gamma_1":0.241,
    "gamma_2":0.291,
    "gamma_3":0.266
}
search_space = {
    "gamma_1":{"type": "uniform", "low": 0.25, "high": 1.5},
    "gamma_2":{"type": "uniform", "low": 0.0, "high": 1},
    "gamma_3":{"type": "uniform", "low": 0.0, "high": 1}
}

trials = 500

for j in range(trials):
    cfg = base_config.copy()
    cfg.update(sample_config(search_space))
    print(cfg)

    training_params['gamma'] = [cfg['gamma_1'],cfg['gamma_2'],cfg['gamma_3'], 0]
    # training_params['gamma'] = [0,0,0, 0]

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



    train_loader = DataLoader(train_data, batch_size, shuffle=True)  # shuffling for minibatch gradient descent
    val_loader = DataLoader(val_data, batch_size, shuffle=False)  # there is no need to shuffle the validation data
    optimizer = torch.optim.AdamW(dynamic_deephit_model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # because we want to minimize validation loss
        factor=0.5,      # multiply lr by 0.5 each time
        patience=8,      # wait 5 epochs of no improvement before reducing
        threshold=1e-3,
        threshold_mode='abs',
        min_lr= 3e-5
    )

    train_epoch_losses = []
    val_sel_epoch_losses = []
    val_dom_epoch_losses = []
    val_nll_epoch_losses = []
    train_nll_epoch_losses = []
    best_val_loss = float('inf')
    best_val = float('inf')
    best_state = None



    dl1_pct_violate = []
    dl2_pct_violate = []
    dl3_pct_violate = []


    model_loss_epoch = None
    domain_loss_epoch = None
    model_losses = []
    domain_losses = []


    for epoch in range(num_epochs):
        # training_params['gamma'] = warmup_gammas(epoch,(cfg['gamma_1'],cfg['gamma_2'],cfg['gamma_3'], 0),warmup_epochs=25)
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
            # --- Train loss ---
            dl1_violations, dl2_violations, dl3_violations = 0.0, 0.0, 0.0
            dl1_total_points, dl2_total_points, dl3_total_points = 0.0, 0.0, 0.0
            tr_log, tr_n = {"model":0,"domain":0,"longit":0,"rank":0,"nll":0,"total":0}, 0
            for Xb, Yb, Db in train_loader:
                Xb = Xb.to(device)
                loss_val, parts = total_loss(dynamic_deephit_model, Xb, Yb, Db, training_params,
                                            for_selection=False, compute_parts=True)
                current_batch_size = Xb.size(0)
        
                for k in tr_log: 
                    tr_log[k] += parts[k] * current_batch_size
                tr_n += current_batch_size


                # dl1_violations += float(parts['dl1_violations'][0])
                # dl1_total_points+= float(parts['dl1_violations'][1])
                # dl2_violations += float(parts['dl2_violations'][0])
                # dl2_total_points+= float(parts['dl2_violations'][1])
                # dl3_violations += float(parts['dl3_violations'][0])
                # dl3_total_points+= float(parts['dl3_violations'][1])


            for k in tr_log: 
                tr_log[k] /= tr_n


            # dl1_pct_violate.append(dl1_violations/dl1_total_points)
            # dl2_pct_violate.append(dl2_violations/dl2_total_points)
            # dl3_pct_violate.append(dl3_violations/dl3_total_points)
            
            train_epoch_losses.append(tr_log['total'])
            train_nll_epoch_losses.append(tr_log['nll'])
            model_losses.append(tr_log['model'])
            domain_losses.append(tr_log['domain'])


            # ---Validation---
            va_log, va_sel,va_dom, va_n = {"domain":0,"longit":0,"rank":0,"nll":0,"total":0}, 0.0, 0.0, 0
            for Xb, Yb, Db in val_loader:
                Xb = Xb.to(device)
                # Compute once; we'll pick out the parts we want
                sel_loss, parts = total_loss(dynamic_deephit_model, Xb, Yb, Db, training_params,
                                    for_selection=True, compute_parts=True)
                _, dom_parts = total_loss(dynamic_deephit_model, Xb, Yb, Db, training_params,
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
            val_sel_epoch_losses.append(va_sel)
            val_dom_epoch_losses.append(va_dom)
        # scheduler.step(val_nll)
        

        print(f"[{epoch+1}] train_total={tr_log['total']:.4f} "
            f"model_loss={tr_log['model']:.4f}  domain_loss={tr_log['domain']:.4f} "
            f"(nll={tr_log['nll']:.4f}, rank={tr_log['rank']:.4f}, longit={tr_log['longit']:.4f})")

        # checkpoint on validation selection metric ONLY
        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in dynamic_deephit_model.state_dict().items()}


    if best_state is not None:
        dynamic_deephit_model.load_state_dict(best_state)

    # fig, ax = plt.subplots()
    # plt.plot(model_losses, label='Model Loss')
    # plt.plot(domain_losses,'--', label='Domain Loss')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Loss')
    # ax.set_title('Model and Domain Loss')
    # ax.set_ylim(0,5)
    # ax.grid()
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(f'/home/arnott/git/CKD-JA-MMSc-1/Results/ParameterExperiment/Real_eGFR/{gamma_1}_{gamma_2}_{gamma_3}_Loss_plot_run{run}.png')
    # pd.Series(model_losses).to_csv(f'/home/arnott/git/CKD-JA-MMSc-1/Results/ParameterExperiment/Real_eGFR/{gamma_1}_{gamma_2}_{gamma_3}_model_losses__run{run}.csv')
    # pd.Series(domain_losses).to_csv(f'/home/arnott/git/CKD-JA-MMSc-1/Results/ParameterExperiment/Real_eGFR/{gamma_1}_{gamma_2}_{gamma_3}_domain_losses_run{run}.csv')

    
    X_test_padded = torch.from_numpy(_get_padded_features(X_test_list)).type(torch.float32).to(device)

    with torch.no_grad():
        yhat, pmf_test = dynamic_deephit_model(X_test_padded)

    cifs = []
    for event_idx_minus_one in range(len(pmf_test)):
        cifs.append(pmf_test[event_idx_minus_one].cumsum(1))
    cif_test_np = np.array([cifs[event_idx_minus_one].cpu().numpy().T for event_idx_minus_one in range(len(events))])
    cif_test_np = cif_test_np.tolist()
    cif_test_np = np.array(cif_test_np)

    Y_train_np = np.array([_[-1] for _ in Y_train_list])
    D_train_np = np.array([_[-1] for _ in D_train_list])
    Y_test_np = np.array([_[-1] for _ in Y_test_list])
    D_test_np = np.array([_[-1] for _ in D_test_list])

    # Loss Evaluation
    t = torch.tensor(Y_test_np).long()
    e = torch.tensor(D_test_np).int()
    cif = [torch.cumsum(ok, dim=1) for ok in pmf_test]
    longit_loss = longitudinal_loss(yhat, X_test_padded)
    rank_loss   = ranking_loss(cif, t, e,  1)
    test_nll_loss    = negative_log_likelihood(pmf_test, cif, t, e)
    test_total_loss = to_float(test_nll_loss) + to_float(rank_loss) + to_float(longit_loss)
    test_nll_loss = to_float(test_nll_loss)
 

    ##### Brier    
    duration_grid_test_np = np.unique(Y_test_np)
    eval_duration_indices = [int(p * len(duration_grid_test_np)) for p in [0.2,0.4,0.6,0.8]]

    censoring_kmf = lifelines.KaplanMeierFitter()
    censoring_kmf.fit(Y_train_np, 1 * (D_train_np == 0))
    brier_scores = {event:[] for event in events}

    for event_idx_minus_one, event in enumerate(events):
        for k, eval_duration_index in enumerate(eval_duration_indices):
            eval_duration = duration_grid_test_np[eval_duration_index]

            # find the training time grid's time point closest to the evaluation time
            interp_time_index = np.argmin(np.abs(eval_duration - duration_grid_train_np))
            cif_values_at_eval_duration_np = cif_test_np[event_idx_minus_one, interp_time_index, :].T
            brier = compute_brier_competing(cif_values_at_eval_duration_np, censoring_kmf, Y_test_np, D_test_np, event_idx_minus_one + 1, eval_duration)

            if k%5==0:
                print(f'Event "{event}" - eval time {eval_duration} - Brier score: {brier}')
            brier_scores[event].append(brier)



    ##### C index  
    # convert training labels into the structured array format used by scikit-survival
    labels_train_sksurv = Surv.from_arrays(1*(D_train_np >= 1), Y_train_np)
    concordance_scores = {event:[] for event in events }

    for event_idx_minus_one, event in enumerate(events):
        # convert test labels into the structured array format used by scikit-survival
        labels_test_sksurv = Surv.from_arrays(1*(D_test_np == (event_idx_minus_one + 1)), Y_test_np)

        for k, eval_duration_index in enumerate(eval_duration_indices):
            eval_duration = duration_grid_test_np[eval_duration_index]

            # find the training time grid's time point closest to the evaluation time
            interp_time_index = np.argmin(np.abs(eval_duration - duration_grid_train_np))
        
            cif_values_at_eval_duration_np = cif_test_np[event_idx_minus_one, interp_time_index, :].T

            concordance = concordance_index_ipcw(labels_train_sksurv, labels_test_sksurv, cif_values_at_eval_duration_np, tau=eval_duration)[0]

            if k%5 == 0:
                print(f'Event "{event}" - eval time {eval_duration} - truncated time-dependent concordance: {concordance}')
            concordance_scores[event].append(concordance)

    brier_death_bool =  (np.array(base_brier_death) >= np.array(brier_scores['death'])).sum() == len(brier_scores['death'])
    brier_dialysis_bool =  (np.array(base_brier_dialysis) >= np.array(brier_scores['dialysis'])).sum() == len(brier_scores['dialysis'])

    if brier_death_bool and brier_dialysis_bool:
        with open(f'/home/arnott/git/CKD-JA-MMSc-1/Results/Competing/Config_run_{run}_{j}.json', "w") as f:
            cfg['brier'] = brier_scores

            json.dump(cfg, f, indent=2)


    cindex_death_bool =  (np.array(base_cindex_death) <= np.array(concordance_scores['death'])).sum() == len(concordance_scores['death'])
    cindex_dialysis_bool =  (np.array(base_cindex_dialysis) <= np.array(concordance_scores['dialysis'])).sum() == len(concordance_scores['dialysis'])

    if cindex_death_bool and cindex_dialysis_bool:
        with open(f'/home/arnott/git/CKD-JA-MMSc-1/Results/Competing/Config_run_{run}_{j}.json', "w") as f:
            cfg['cindex'] = concordance_scores
            json.dump(cfg, f, indent=2)



        