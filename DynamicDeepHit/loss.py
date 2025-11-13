'''
This file is by Vincent Jeanselme and was downloaded from:
    https://github.com/Jeanselme/DynamicDeepHit/blob/0c160824d37d6f5758e7b65d8e566aefac0901e9/ddh/losses.py
'''
import torch
import torch.nn.functional as F
from DynamicDeepHit.ddh_torch import DynamicDeepHitTorch

def negative_log_likelihood(outcomes, cif, t, e):
    """
    Returns mean NLL per sample (not per cause, not summed).
    """
    total = cif[0].new_tensor(0.0)
    count = cif[0].new_tensor(0.0)

    for k, ok in enumerate(outcomes):
        # uncensored for cause k+1
        sel = (e == (k + 1))
        if sel.any():
            # ok[sel]: [num_sel, T]; pick prob at observed time t[sel]
            probs = ok[sel][torch.arange(sel.sum()), t[sel]]
            total += torch.sum(torch.log(torch.clamp(probs, min=1e-10)))
            count += sel.sum()

    # censored contribution (uses sum over causes' CIFs)
    cens_mask = (e == 0)
    if cens_mask.any():
        cens_cif = 0
        for k in range(len(outcomes)):
            cens_cif = cens_cif + cif[k][cens_mask][torch.arange(cens_mask.sum()), t[cens_mask]]
        cens_input = 1.0 - cens_cif
        total += torch.sum(torch.log(torch.clamp(cens_input, min=1e-10)))
        count += cens_mask.sum()

    # mean over samples that contributed
    count = torch.clamp(count, min=1.0)
    return - total / count


def ranking_loss(cif, t, e, sigma):
    """
    Mean over all valid (i,j) comparisons across causes.
    """
    loss_sum = cif[0].new_tensor(0.0)
    n_terms  = cif[0].new_tensor(0.0)

    for k, cifk in enumerate(cif):
        mask_k = (e - 1 == k)                  # subjects with event k at some time
        cifk_k = cifk[mask_k]                  # [Nk, T]
        t_k    = t[mask_k]                     # [Nk]

        for ci, ti in zip(cifk_k, t_k):
            later = (t > ti)
            if later.any():
                # probability of event k at time ti for those still at risk
                diff = cifk[later][torch.arange(later.sum()), ti] - ci[ti]
                term = torch.exp(diff / sigma)
                loss_sum += term.mean()
                n_terms  += 1

    n_terms = torch.clamp(n_terms, min=1.0)
    return loss_sum / n_terms


def longitudinal_loss(longitudinal_prediction, x,kappa):
    """
    Penalize error in the longitudinal predictions
    This function is used to compute the error made by the RNN

    NB: In the paper, they seem to use different losses for continuous and categorical
    But this was not reflected in the code associated (therefore we compute MSE for all)

    NB: Original paper mentions possibility of different alphas for each risk
    But take same for all (for ranking loss)
    """

    length = (~torch.isnan(x[:,:,0])).sum(dim = 1) - 1
    device = x.device
    last_index = length.clamp(min=0)
    # Create a grid of the column index
    # index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)
    index = torch.arange(x.size(1), device=device).unsqueeze(0)  # [1, T]

    # Select all predictions until the last observed
    # prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))
    pred_last = (last_index - 1).clamp(min=-1)                     # -1 => no predictions
    prediction_mask = index <= pred_last.unsqueeze(1)
    # print(f'predict mask: {longitudinal_prediction[prediction_mask]}')

    # Select all observations that can be predicted
    observation_mask = index <= last_index.unsqueeze(1)            # [B, T]
    observation_mask[:, 0] = False

    pred_T = longitudinal_prediction.size(1)
    prediction_mask = prediction_mask[:, :pred_T]
    obs_T = x.size(1)
    observation_mask = observation_mask[:, :obs_T]


    X_mat = x[observation_mask]
    Y_mat = longitudinal_prediction[prediction_mask]
    valid_mask = (X_mat != -1) & (Y_mat != -1)
    X_valid = X_mat[valid_mask]
    Y_valid = Y_mat[valid_mask]

    abs_err = (X_valid - Y_valid).abs()
    huber = torch.where(abs_err < kappa, 0.5*abs_err**2, kappa*(abs_err - 0.5*kappa))
    loss = huber.mean()    
    return loss


def domain_loss_1(longitudinal_prediction, x, tol = 1):
    """Compute the eGFR equation for the loss"""
    device = x.device

    # Determine sequence lengths (number of observed time steps)
    # length = (~torch.isnan(x[:,:,0])).sum(axis = 1) - 1
    length = (~torch.isnan(x[:,:,0])).sum(axis = 1) - 1
    index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)

    # Masks for observations and predictions    
    prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))
    observation_mask = index <= length.unsqueeze(1).repeat(1, x.size(1))
    observation_mask[:, 0] = False # Remove first observation

    # Slice data
    gender = x[:,:,0]
    age = x[:,:,1]
    predicted_eGFR = longitudinal_prediction[:,:,4]
    predicted_Cr = longitudinal_prediction[:,:,2]
        
    ## Apply masks
    gender_obs = gender[observation_mask]
    age_obs = age[observation_mask]
    predicted_eGFR_obs = predicted_eGFR[prediction_mask]
    predicted_Cr_obs = predicted_Cr[prediction_mask]
    
    ## Missing mask
    valid_mask = (gender_obs != -1) & (age_obs != -1) & (predicted_eGFR_obs != -1) & (predicted_Cr_obs != -1)
    gender_masked = gender_obs[valid_mask]
    age_masked = age_obs[valid_mask]
    predicted_eGFR_masked = predicted_eGFR_obs[valid_mask]
    predicted_Cr_masked = predicted_Cr_obs[valid_mask]

    ## Gender dependent flags
    k = torch.where(gender_masked==1, 0.7, 0.9)
    a = torch.where(gender_masked==1, -0.241, -0.302)
    gender_flag = torch.where(gender_masked==1, 1.012, 1)

    # Min max terms
    creatine_over_k = predicted_Cr_masked/ k
    creatine_over_k = torch.clamp(creatine_over_k, min=1e-5)
    ones = torch.ones_like(creatine_over_k)
    min_term = torch.minimum(creatine_over_k,ones) # causing problems (small -ve number with negative exponent)
    max_term = torch.maximum(creatine_over_k,ones) # well behaved

    # print(f'predicted creatine: {torch.isnan(predicted_Cr_masked).any()}') # no nans
    # print(f'Creatine over k: {creatine_over_k}')
    # print(f'min term: {min_term}')
    # print(f'max term: {max_term}')

    # Age and calculation
    age_term = 0.9938**age_masked
    calculated_eGFR = 142 * (min_term**a) * (max_term**(-1.2)) * age_term * gender_flag

    squared_error = F.mse_loss(calculated_eGFR, predicted_eGFR_masked, reduction='none') 
    abs_err = (predicted_eGFR_masked - calculated_eGFR).abs()
    # print(abs_err)
    within = abs_err> tol
    hits = within.sum()

    dl1_loss = squared_error.mean() 
    return dl1_loss, (hits, abs_err.numel())

def domain_loss_2(longitudinal_prediction, bound_dict): ##TODO - Find some upper and lower bounds
    """A_indices: Set of indicies for bounded variables"""
    # longitudinal_prediction shape
    bound_dict ={int(k): v for k, v in bound_dict.items()}
    bound_indices = list(bound_dict.keys())
    preds = longitudinal_prediction[:, :, bound_indices]
    
    # lower bound loss
    lower_bounds = torch.tensor([bound_dict[i][0] for i in bound_indices],device=preds.device, dtype=preds.dtype)                           
    lower_bounds_broadcast = lower_bounds.reshape((1, 1, -1))
    lower_loss = torch.relu(lower_bounds_broadcast - preds)
    return lower_loss.mean(), (torch.count_nonzero(lower_loss),lower_loss.numel())

def domain_loss_3(longitudinal_prediction, bound_dict): ##TODO - Find some upper and lower bounds
    """A_indices: Set of indicies for bounded variables"""
    # longitudinal_prediction shape
    bound_dict = {int(k): v for k, v in bound_dict.items()}
    bound_indices = list(bound_dict.keys())
    preds = longitudinal_prediction[:, :, bound_indices]

    #Upper bound loss
    upper_bounds = torch.tensor([bound_dict[i][1] for i in bound_indices],device=preds.device, dtype=preds.dtype)                           
    upper_bounds_broadcast = upper_bounds.reshape((1, 1, -1))
    upper_loss = torch.relu(preds - upper_bounds_broadcast)
    return upper_loss.mean(), (torch.count_nonzero(upper_loss),upper_loss.numel())


def domain_loss_4(longitudinal_prediction, x,option = 1):
    ##TODO - Implement
    if x.is_cuda:
        device = x.get_device()
    else:
        device = torch.device("cpu")
        
    if option == 1:
        pass
    
    return torch.tensor(0.0, dtype=torch.float, device=device)

def total_loss(
    model:DynamicDeepHitTorch, x, t, e, training_params:dict, eval_flag: bool = False) -> float | dict:
    """
    Returns (loss_scalar, parts_dict) if compute_parts else (loss_scalar, None).
    'loss_scalar' is:
      - TRAIN: model loss + domain penalties (when use_constraints==True and for_selection=False)
      - VAL selection: model loss ONLY (domain excluded), unless you explicitly include it.

    Notes:
      - Weights: longit_w = max(0, 1 - alpha - beta) to avoid accidental negative weights.
      - Domain terms are only computed when needed to save time.
    """
    # Unpack parameters
    alpha = training_params['alpha']   # rank weight
    beta  = training_params['beta']    # NLL weight
    delta = training_params['delta']
    sigma = training_params['sigma']
    kappa = training_params['kappa']
    gamma_1, gamma_2, gamma_3 = training_params['gamma']
    bound_dict = training_params['bound_dict']
    use_constraints = training_params.get('use_constraints', False)

    # forward
    longitudinal_prediction, outcomes = model(x)
    t, e = t.long(), e.int()

    # heads → CIFs
    cif = [torch.cumsum(ok, dim=1) for ok in outcomes]

    # core losses
    longit_loss = longitudinal_loss(longitudinal_prediction, x,kappa)
    rank_loss   = ranking_loss(cif, t, e, sigma)
    nll_loss    = negative_log_likelihood(outcomes, cif, t, e)

    # ensure non-negative combination weights
    model_loss = delta * longit_loss + alpha * rank_loss + beta * nll_loss

    if not use_constraints:
        if not eval_flag:
            return model_loss
        return {
            "nll" : nll_loss, 
            "rank" : rank_loss,
            "longit" : longit_loss,
            "dl1": 0,
            "dl2": 0,
            "dl3": 0,
            }

    dl1, _ = domain_loss_1(longitudinal_prediction, x)
    dl2, _ = domain_loss_2(longitudinal_prediction, bound_dict)
    dl3, _ = domain_loss_3(longitudinal_prediction, bound_dict)
    domain_loss = gamma_1 * dl1 + gamma_2 * dl2 + gamma_3 * dl3
    if not eval_flag:
        return model_loss + domain_loss
    
    return {
            "nll" : nll_loss, 
            "rank" : rank_loss,
            "longit" : longit_loss,
            "dl1": dl1,
            "dl2": dl2,
            "dl3": dl3,
            }
    
    

            














"""
    need_domain = (use_constraints and not for_selection)
    if need_domain:
        if detach_domain_from_trunk:
            # optional: prevent domain gradients from flowing back into the shared trunk
            lp = longitudinal_prediction.detach()
        else:
            lp = longitudinal_prediction

        dl1, dl1_violations = domain_loss_1(lp, x)
        dl2, dl2_violations = domain_loss_2(lp, bound_dict)
        dl3, dl3_violations = domain_loss_3(lp, bound_dict)

        # Optional: clip or huberise individual dl* here if you find occasional spikes
        domain_loss = gamma_1 * dl1 + gamma_2 * dl2 + gamma_3 * dl3

    if not use_constraints:
        lp = longitudinal_prediction
        dl1, dl1_violations = domain_loss_1(lp, x)
        dl2, dl2_violations = domain_loss_2(lp, bound_dict)
        dl3, dl3_violations = domain_loss_3(lp, bound_dict)

    # For selection on validation: typically exclude domain
    if for_selection:
        # If you prefer NLL-only selection, set include_rank_in_selection=False
        sel_loss = (delta * longit_loss) + (alpha * rank_loss if include_rank_in_selection else 0.0) + beta * nll_loss
        loss_scalar = sel_loss
    else:
        loss_scalar = model_loss + domain_loss

    parts = None
    if compute_parts:
        parts = {
            "longit": float(longit_loss),
            "rank":   float(rank_loss),
            "nll":    float(nll_loss),
            "model":  float(model_loss),
            "dl1":    float(dl1) if need_domain else 0.0,
            "dl2":    float(dl2) if need_domain else 0.0,
            "dl3":    float(dl3) if need_domain else 0.0,
            "dl1_violations": dl1_violations if need_domain  else 0.0,
            "dl2_violations": dl2_violations if need_domain  else 0.0,
            "dl3_violations": dl3_violations if need_domain  else 0.0,
            "domain": float(domain_loss) if need_domain else 0.0,
            "total":  float(loss_scalar),
        }

    return loss_scalar, parts

"""