import json
import numpy as np
import pandas as pd
import torch
import lifelines
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw

def load_data(file_name):
    """Load In results"""
    with open(file_name,'r') as f:
        data = json.load(f)
    return data

def compute_brier_competing(cif_values_at_time_horizon, censoring_kmf,
                            Y_test, D_test, event_of_interest, time_horizon):
    n = len(Y_test)
    assert len(D_test) == n

    residuals = np.zeros(n)
    for idx in range(n):
        observed_time = Y_test[idx]
        event_indicator = D_test[idx]
        if observed_time > time_horizon:
            weight = censoring_kmf.predict(time_horizon)
            residuals[idx] = (cif_values_at_time_horizon[idx])**2 / weight
        else:
            weight = censoring_kmf.predict(observed_time)
            if event_indicator == event_of_interest:
                residuals[idx] = (1 - cif_values_at_time_horizon[idx])**2 / weight
            elif event_indicator != event_of_interest and event_indicator != 0:
                residuals[idx] = (cif_values_at_time_horizon[idx])**2 / weight
    return residuals.mean()

def discretize(t, split, split_time=None):
    """
    Discretize the survival horizon.

    Args:
        t (List[np.ndarray]): times per sequence
        split (int): number of bins desired
        split_time (np.ndarray | None): bin edges to reuse

    Returns:
        (np.ndarray[object], np.ndarray): indices per sequence, bin edges
    """
    if split_time is None:
        # Use exactly `split` bins → edges has length split+1
        _, edges = np.histogram(np.concatenate(t), bins=split)
    else:
        edges = np.asarray(split_time)

    # left-closed, right-open: bins[i] <= x < bins[i+1]
    idx = [np.digitize(t_, edges, right=False) - 1 for t_ in t]

    # clip under/overflow into first/last valid bin
    nbins = len(edges) - 1
    idx = [np.clip(i, 0, nbins - 1) for i in idx]

    return np.array(idx, dtype=object), edges

def zip_features(X,Y,D, device):
    X_padded = torch.from_numpy(_get_padded_features(X)).type(torch.float32).to(device)
    last_Y_train = torch.tensor([_[-1] for _ in Y]).type(torch.float32).to(device)
    D_train = torch.tensor([_[-1] for _ in D]).type(torch.float32).to(device)
    data = list(zip(X_padded, last_Y_train, D_train))
    return data, X_padded



def _get_padded_features(x):
    """Helper function to pad variable length RNN inputs with nans."""
    d = max([len(x_) for x_ in x])
    padx = []
    for i in range(len(x)):
        pads = np.nan*np.ones((d - len(x[i]),) + x[i].shape[1:])
        padx.append(np.concatenate([x[i], pads]))
    return np.array(padx)


def interleave_subjects_by_state(df, subject_col='subject_id', state_col='state'):
    # Step 1: Get one row per subject (to capture subject-state mapping)
    subject_state_df = df.drop_duplicates(subset=subject_col)[[subject_col, state_col]]

    # Step 2: Group subjects by state
    state_groups = {
        state: list(group[subject_col])
        for state, group in subject_state_df.groupby(state_col)
    }
    # Step 3: Round-robin subject ordering
    interleaved_subjects = []
    max_len = max(len(v) for v in state_groups.values())
    states = sorted(state_groups.keys())  # optional: change ordering if needed

    for i in range(max_len):
        for state in states:
            subjects = state_groups[state]
            if i < len(subjects):
                interleaved_subjects.append(subjects[i])

    # Step 4: Reconstruct full time-series DataFrame
    df = df.set_index(subject_col)
    ordered_df = pd.concat([df.loc[[sid]] for sid in interleaved_subjects])
    return ordered_df.reset_index()

def subset_data(X, Y, D, N,seed):
    np.random.seed(seed)
    max_int = len(X)
    indicies  = np.random.choice(np.arange(0, max_int), size=N, replace=False)
    X = [X[i] for i in indicies]
    Y = [Y[i] for i in indicies]
    D = [D[i] for i in indicies]
    return X,Y,D


def compute_brier(Y_train_np, Y_test_np, D_train_np, D_test_np,\
                   events, duration_grid_train_np, cif_test_np,\
                    eval_duration_indices,duration_grid_test_np):
        

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
        return brier_scores

def compute_cindex(Y_train_np, Y_test_np, D_train_np, D_test_np,\
                   events, duration_grid_train_np, cif_test_np,\
                    eval_duration_indices, duration_grid_test_np):
    labels_train_sksurv = Surv.from_arrays(1*(D_train_np >= 1), Y_train_np)
    concordance_scores = {event:[] for event in events }

    for event_idx_minus_one, event in enumerate(events):
        # convert test labels into the structured array format used by scikit-survival
        try:
            labels_test_sksurv = Surv.from_arrays(1*(D_test_np == (event_idx_minus_one + 1)), Y_test_np)
        except:
            concordance_scores[event].append(None)
            break
        for k, eval_duration_index in enumerate(eval_duration_indices):
            eval_duration = duration_grid_test_np[eval_duration_index]

            # find the training time grid's time point closest to the evaluation time
            interp_time_index = np.argmin(np.abs(eval_duration - duration_grid_train_np))
        
            cif_values_at_eval_duration_np = cif_test_np[event_idx_minus_one, interp_time_index, :].T

            concordance = concordance_index_ipcw(labels_train_sksurv, labels_test_sksurv, cif_values_at_eval_duration_np, tau=eval_duration)[0]

            if k%5 == 0:
                print(f'Event "{event}" - eval time {eval_duration} - truncated time-dependent concordance: {concordance}')
            concordance_scores[event].append(concordance)

    return concordance_scores

def to_float(x):
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    return float(x)
