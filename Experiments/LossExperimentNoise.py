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
from ddh.losses_CRexp2 import total_loss, ranking_loss,longitudinal_loss, negative_log_likelihood
from utils import discretize, _get_padded_features, load_data, compute_brier_competing, subset_data
from copy import deepcopy
import torch

def normalize_features(X_list, fit_stats=None):
    """Normalize features across all time steps"""
    # Concatenate all features across patients and time steps
    all_features = np.concatenate([x for x in X_list], axis=0)
    
    if fit_stats is None:
        # Compute stats on training data
        means = np.mean(all_features, axis=0)
        stds = np.std(all_features, axis=0)
        stds[stds == 0] = 1  # Avoid division by zero
        fit_stats = {'means': means, 'stds': stds}
    
    # Apply normalization
    X_normalized = []
    for x in X_list:
        x_norm = (x - fit_stats['means']) / fit_stats['stds']
        # Don't normalize binary gender feature
        x_norm[:, 0] = x[:, 0]  
        X_normalized.append(x_norm)
    
    return X_normalized, fit_stats

def sample_patients(df,n=1000, id_col='subject_id',random_state=None):
    ids = df[id_col]
    unique_ids = ids.drop_duplicates()
    sampled_ids = unique_ids.sample(n=n,random_state =random_state)
    return df[df[id_col].isin(sampled_ids)]
import numpy as np

def add_noise(data, sigma=0.05, rng=1111):
    rng = np.random.default_rng(rng)
    noisy = data.copy()

    for col in [6,7,8,9,10]:
        x = data.iloc[:, col]
        noise = rng.normal(0, sigma * np.std(x), size=len(x))
        noisy.iloc[:, col] = x + noise
    return noisy


def to_float(x):
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    return float(x)

cuda = 'cuda'
dataset = '/home/arnott/git/CKD-JA-MMSc-1/Data/real_eGFR_mimic.csv'


features_before_preprocessing = ['gender','age_calculated','Creatinine','Hemoglobin','eGFR','UPCR','ACUR']
events = ['death']

stds = [0.05, 0.1, 0.5, 1, 2]
df = pd.read_csv(dataset)
noise_nll_scores = []

df[df.isna()] = -1 ##NOTE - Mask missing features with -1
df['gender'] = np.where(df['gender']=='F',1,0) # female = 1, male = 0
df = df[df['subject_id'].map(df['subject_id'].value_counts()) > 1] # remove tabular data
df = df.sort_values(['subject_id', 'year'], ascending=[True, True]).reset_index(drop=True)

for std in stds:
    noisy_df = add_noise(df,std)

    features = noisy_df[features_before_preprocessing].to_numpy().astype('float32')    
    observed_times = (noisy_df['years'] - noisy_df['year']).to_numpy().astype('float32') # Calculate Time to Events
    event_indicators = (noisy_df['event'] == 'death').to_numpy().astype('int32') # 0 == survived (censored), 1 == death, 2== dialysis
    event_indicators[noisy_df['event'] == 'dialysis'] = 2
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

    num_durations = 512  # set this to 0 to use all unique durations in which any critical event happens

    # discretize training durations
    Y_train_discrete_np, duration_grid_train_np = discretize(Y_train_list, num_durations)

    # discretize validation observed durations using the grid obtained based on training data
    Y_val_discrete_np, _ = discretize(Y_val_list, len(duration_grid_train_np) - 1, duration_grid_train_np)

    output_num_durations = len(duration_grid_train_np)
    # print(f'Number of discretized durations to be used with Dynamic-DeepHit: {output_num_durations}')
    # print('Duration grid:', duration_grid_train_np)


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

    # RNN parameters
    num_hidden = 96  # RNN hidden layer number of nodes (i.e., dimension of RNN hidden layer); `width` [8,16,32,64]
    num_rnn_layers = 2 # number of lstm layers `depth` [2,4,8]
    rnn_type = 'LSTM'  # 'RNN', 'LSTM', and 'GRU'
    dropout = 0.22300435446772132

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
    run = 3
    use_constraints = False
    training_params =  {"alpha": 0.15598768146289432, # rank loss
                        "beta":  1, # NLL loss 
                        "delta": 0.08538145469419971,   # longit loss
                        "sigma": 0.35434828698561155,
                        "kappa":1.5,
                        # "gamma": [0,0,0, 0],
                        # "gamma": [0.32850085347427554,0.6463939029213741,0.6227217068994422, 0],
                        "gamma":[0.8184331463486942,1.4071753875280206,0.15588194083088636,0],
                        "bound_dict": {3: [0,17.5],  # Cr 2: [0,17.0]    0.336,0.026,0.181,        
                                    4: [0,128.3014697269145],
                                    5: [0,17.6],
                                    6: [0,12767]
                                    },
                        "use_constraints":use_constraints,
                        }

    num_epochs = 10000
    batch_size = 64 # 128,256,512,1028
    learning_rate = 0.22300435446772132
    grad_clip = 0.5
    best_avg = float('inf')



    train_loader = DataLoader(train_data, batch_size, shuffle=True)  # shuffling for minibatch gradient descent
    val_loader = DataLoader(val_data, batch_size, shuffle=False)  # there is no need to shuffle the validation data
    optimizer = torch.optim.AdamW(dynamic_deephit_model.parameters(), lr=learning_rate, weight_decay=1e-3)
    warmup_steps = 100
    global_step = 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,              
        patience=25,           #No of Epochs with no improvement after which LR reduced   
        threshold= 5.35868151941114e-05,         #
        threshold_mode='rel',
        cooldown=5, # Number of epochs to wait before resuming normal after reduction
        min_lr= 4.542040611324688e-06
    )

    train_epoch_losses = []
    val_sel_epoch_losses = []
    val_dom_epoch_losses = []
    val_nll_epoch_losses = []
    train_nll_epoch_losses = []
    val_longit_epoch_losses = []

    dl1_pct_violate = []
    dl2_pct_violate = []
    dl3_pct_violate = []

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
            if global_step < warmup_steps:
                warmup_lr = 1e-6 + (1e-3 - 1e-6) * (global_step / warmup_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr
            
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
            dl1_violations, dl2_violations, dl3_violations = 0.0, 0.0, 0.0
            dl1_total_points, dl2_total_points, dl3_total_points = 0.0,0.0,0.0
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
                # dl1_total_points += float(parts['dl1_violations'][1])

                # dl2_violations += float(parts['dl2_violations'][0])
                # dl2_total_points += float(parts['dl2_violations'][1])

                # dl3_violations += float(parts['dl3_violations'][0])
                # dl3_total_points += float(parts['dl3_violations'][1])


            for k in tr_log: 
                tr_log[k] /= tr_n

            # dl1_pct_violate.append(dl1_violations/dl1_total_points)
            # dl2_pct_violate.append(dl2_violations/dl2_total_points)
            # dl3_pct_violate.append(dl3_violations/dl3_total_points)

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
            val_longit_epoch_losses.append(va_log['longit'])
        # scheduler.step(val_nll)
        

        print(f"[{epoch+1}] model_loss={tr_log['model']:.4f} "
            f"domain_loss={tr_log['domain']:.4f}  val_domain={va_log['domain']:.4f} "
            f"(nll={va_log['nll']:.4f}, rank={va_log['rank']:.4f}, longit={va_log['longit']:.4f})")
        
        

        # print(f"[{epoch+1}] train_total={tr_log['total']:.4f} "
        #       f"val_sel={val_nll:.4f}  val_domain={va_log['domain']:.4f} "
        #       f"(nll={tr_log['nll']:.4f}, rank={tr_log['rank']:.4f}, longit={tr_log['longit']:.4f})")

        # checkpoint on validation selection metric ONLY
        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in dynamic_deephit_model.state_dict().items()}

        # current_avg = rolling_average(val_sel_epoch_losses,window = 30)
        # if current_avg < best_avg - eps:
        #     best_avg = current_avg
        #     bad_streak = 0
        # else:
        #     bad_streak +=1
        #     if bad_streak > patience:
        #         print("Stopping Early; no improvement on validation")
        #         break

    # restore best
    if best_state is not None:
        dynamic_deephit_model.load_state_dict(best_state)


    X_test_padded = torch.from_numpy(_get_padded_features(X_test_list)).type(torch.float32).to(device)
    Y_train_np = np.array([_[-1] for _ in Y_train_list])
    D_train_np = np.array([_[-1] for _ in D_train_list])
    Y_test_np = np.array([_[-1] for _ in Y_test_list])
    D_test_np = np.array([_[-1] for _ in D_test_list])
    with torch.no_grad():
        yhat, pmf_test = dynamic_deephit_model(X_test_padded)

    t = torch.tensor(Y_test_np).long()
    e = torch.tensor(D_test_np).int()
    cif = [torch.cumsum(ok, dim=1) for ok in pmf_test]
    longit_loss = (longitudinal_loss(yhat, X_test_padded,kappa=training_params['kappa']))
    rank_loss   = ranking_loss(cif, t, e,  1)
    test_nll_loss    = to_float(negative_log_likelihood(pmf_test, cif, t, e))
    test_longit_loss = to_float(longit_loss)
    test_total_loss = test_nll_loss + to_float(rank_loss) + test_longit_loss
    noise_nll_scores.append(test_nll_loss)
    print(f"Test Total Loss: {test_total_loss}")
    print(f"Test NLL Loss: {test_nll_loss}")
    print(f"Test Longit Loss: {longit_loss}")

    # base_data = load_data('/home/arnott/git/CKD-JA-MMSc-1/Results/LossExperiment/Notebook/base_data_1.json')
    fig, ax = plt.subplots(2,2, figsize = (15,5))
    # ax[0,0].plot(base_data['train_epoch_losses'], label='Train DDH')
    ax[0,0].plot(train_epoch_losses, label='Train CKD')
    ax[0,0].set_xlabel('Epoch')
    ax[0,0].set_ylabel('Loss')
    ax[0,0].set_ylim(0,2)
    ax[0,0].set_title('Training Loss')
    ax[0,0].grid()
    ax[0,0].legend()

    # ax[0,1].plot(base_data['val_sel_epoch_losses'] ,label='validation DDH')
    ax[0,1].plot(val_sel_epoch_losses ,label='validation CKD')

    ax[0,1].set_xlabel('Epoch')
    # ax[1].set_ylim(0,17)
    ax[0,1].set_ylabel('Loss')
    ax[0,1].set_title('Validation Loss')
    ax[0,1].grid()
    ax[0,1].legend()

    # ax[1,0].plot(base_data['val_nll_epoch_losses'] ,label='nll validation DDH')
    ax[1,0].plot(val_nll_epoch_losses ,label='nll validation CKD')

    ax[1,0].set_xlabel('Epoch')
    ax[1,0].set_ylabel('Loss')
    ax[1,0].set_title('NLL Validation Loss')
    ax[1,0].grid()
    ax[1,0].legend()

    # ax[1,1].plot(base_data['val_longit_epoch_losses'] ,label='nll validation DDH')
    ax[1,1].plot(val_longit_epoch_losses ,label='nll validation CKD')

    ax[1,1].set_xlabel('Epoch')
    ax[1,1].set_ylabel('Loss')
    ax[1,1].set_title('Longit Validation Loss')
    ax[1,1].grid()
    ax[1,1].legend()


    plt.tight_layout()
    # plt.savefig(f'/home/arnott/git/CKD-JA-MMSc-1/Results/LossExperiment/TEsting/base_loss_1.png')

    # plt.savefig(f'/home/arnott/git/CKD-JA-MMSc-1/Results/LossExperiment/Notebook/run_{run}_plot.png')


    # data = { "use_constraints":use_constraints,
    #             "test_total_loss": test_total_loss,
    #             "test_NLL_loss": test_nll_loss,
    #             "test_longit_loss": test_longit_loss,
    #             "gamma":training_params['gamma'],
    #             'train_epoch_losses':train_epoch_losses,
    #             'val_sel_epoch_losses':val_sel_epoch_losses,
    #             'val_nll_epoch_losses':val_nll_epoch_losses,
    #             "val_longit_epoch_losses":val_longit_epoch_losses
    #             }
    
    # with open(f'/home/arnott/git/CKD-JA-MMSc-1/Results/LossExperiment/Notebook/run_{run}_params.json', "w") as f:
    #     json.dump(data, f, indent=2)

    with open(f'/home/arnott/git/CKD-JA-MMSc-1/ResultsFinal/RQ3/Noise/base_data.json', "w") as f:     
        json.dump(noise_nll_scores, f, indent=2)
