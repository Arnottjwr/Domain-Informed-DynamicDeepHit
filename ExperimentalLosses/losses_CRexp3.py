
import torch
import torch.nn.functional as F

def clean_longitudinal_1(Y,X):
    # Calculate total loss
    valid = torch.isfinite(Y) & torch.isfinite(X)   # (B, T)

    # elementwise loss
    mse_elem = F.mse_loss(Y, X, reduction='none')  # (B, T)

    # average over valid entries only
    loss = (mse_elem[valid].mean() if valid.any() else torch.tensor(0., device=Y.device))
    return loss


def clean_longitudinal(Y, X, extra_mask=None):
    """
    Returns (loss_sum, count), both scalar tensors.
    loss_sum is the sum over valid positions; count is number of valid positions.
    """
    # base validity: both inputs finite
    valid = torch.isfinite(Y) & torch.isfinite(X)
    if extra_mask is not None:
        valid = valid & extra_mask

    # elementwise squared error
    err2 = (Y - X).pow(2)

    # IMPORTANT: zero-out invalid BEFORE summing, so NaNs don't propagate
    err2 = torch.where(
        valid, 
        err2, 
        torch.zeros((), dtype=err2.dtype, device=err2.device)
    )

    loss_sum = err2.sum()
    count    = valid.sum()

    return loss_sum, count                   # scalar tensors



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


def longitudinal_loss(longitudinal_prediction, x):
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
    delta = 1.0  # tune
    abs_err = (X_valid - Y_valid).abs()
    huber = torch.where(abs_err < delta, 0.5*abs_err**2, delta*(abs_err - 0.5*delta))
    loss = huber.mean()    
    return loss


def domain_loss_1(y, x, tol = 0.1):
    """Compute the eGFR equation for the loss"""
    device = x.device

    # Determine sequence lengths (number of observed time steps)
    length = (~torch.isnan(x[:,:,0])).sum(axis = 1) - 1
    index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)
    prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))
    observation_mask = (index <= length.unsqueeze(1).repeat(1, x.size(1)))
    observation_mask[:,0] = False

    gender = x[:,:,0]
    age = x[:,:,1]
    predicted_eGFR = y[:,:,4]
    predicted_Cr = y[:,:,2]


    # masks you already have (B, T)
    m_obs  = observation_mask           # where covariates are observed
    m_pred = prediction_mask            # where predictions exist

    # per-variable validity (still (B, T))
    m_gender = m_obs  & (gender         != -1)
    m_age    = m_obs  & (age            != -1)
    m_eGFR   = m_pred & (predicted_eGFR != -1)
    m_Cr     = m_pred & (predicted_Cr   != -1)


    # positions where **all** needed values are valid
    valid2d = m_gender & m_age & m_eGFR & m_Cr                     # (B, T)


    # Keep shape: replace invalid entries with NaN (or any fill you prefer)
    fill_g = torch.full_like(gender,         float('nan'))
    fill_a = torch.full_like(age,            float('nan'))
    fill_e = torch.full_like(predicted_eGFR, float('nan'))
    fill_c = torch.full_like(predicted_Cr,   float('nan'))


    gender_masked2d        = torch.where(valid2d, gender,         fill_g)  # (B, T)
    age_masked2d           = torch.where(valid2d, age,            fill_a)
    predicted_eGFR_masked2d= torch.where(valid2d, predicted_eGFR, fill_e)
    predicted_Cr_masked2d  = torch.where(valid2d, predicted_Cr,   fill_c)


    ## Gender specific components
    g = gender_masked2d.to(torch.float32)  # ensure float so 0.7/0.9 aren’t truncated
    k = torch.where(g == 1, torch.tensor(0.7, device=g.device, dtype=g.dtype),
        torch.where(g == 0, torch.tensor(0.9, device=g.device, dtype=g.dtype), g))

    a = torch.where(g == 1, torch.tensor(-0.241, device=g.device, dtype=g.dtype),
        torch.where(g == 0, torch.tensor( -0.302, device=g.device, dtype=g.dtype), g))

    gender_flag = torch.where(g == 1, torch.tensor(1.012, device=g.device, dtype=g.dtype),
        torch.where(g == 0, torch.tensor( 1, device=g.device, dtype=g.dtype), g))


    # Min max terms
    creatine_over_k = predicted_Cr_masked2d/ k
    creatine_over_k = torch.clamp(creatine_over_k, min=1e-5)

    ones = torch.ones_like(creatine_over_k)
    min_term = torch.minimum(creatine_over_k,ones) # causing problems (small -ve number with negative exponent)
    max_term = torch.maximum(creatine_over_k,ones) # well behaved

    # print(f'predicted creatine: {torch.isnan(predicted_Cr_masked).any()}') # no nans
    # print(f'Creatine over k: {creatine_over_k}')
    # print(f'min term: {min_term}')
    # print(f'max term: {max_term}')

    # Age and calculation
    age_term = 0.9938**age_masked2d
    calculated_eGFR = 142 * (min_term**a) * (max_term**(-1.2)) * age_term * gender_flag

    ## Determine violating and compliant points
    
    absolute_val = (predicted_eGFR_masked2d- calculated_eGFR).abs()
    violating_points_mask = absolute_val > tol
    compliant_points_mask = absolute_val <= tol


    mask2dv = violating_points_mask.bool()           # (B, T)
    mask3dv = mask2dv.unsqueeze(-1)           # (B, T, 1) — broadcasts over F
    mask2dc = compliant_points_mask.bool()           # (B, T)
    mask3dc = mask2dc.unsqueeze(-1) 


    # choose a fill value; NaN if floating, else a sentinel like -1
    fill_valy = float('nan') if y.is_floating_point() else -1
    fill_y = torch.full_like(y, fill_valy)

    fill_valx = float('nan') if x.is_floating_point() else -1
    fill_x = torch.full_like(x, fill_valx)

    # Group A: where mask is True; Group B: where mask is False
    violating_points_predictions = torch.where(mask3dv, y, fill_y)      # (B, T, F)
    compliant_points_predictions = torch.where(mask3dc, y, fill_y)     # (B, T, F)

    violating_points_inputs = torch.where(mask3dv, x, fill_x)      # (B, T, F)
    compliant_points_inputs = torch.where(mask3dc, x, fill_x)     # (B, T, F)


    # Calculate total loss
    valid = torch.isfinite(calculated_eGFR) & torch.isfinite(predicted_eGFR_masked2d)   # (B, T)

    # elementwise loss
    mse_elem = F.mse_loss(calculated_eGFR, predicted_eGFR_masked2d, reduction='none')  # (B, T)

    # average over valid entries only
    loss = (mse_elem[valid].mean()                           # scalar
            if valid.any() else torch.tensor(0., device=calculated_eGFR.device))
    return loss, (compliant_points_inputs,compliant_points_predictions), (violating_points_inputs,violating_points_predictions)



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
    return lower_loss.mean()

def domain_loss_3(longitudinal_prediction, bound_dict): ##TODO - Find some upper and lower bounds
    """A_indices: Set of indicies for bounded variables"""
    # longitudinal_prediction shape
    bound_dict ={int(k): v for k, v in bound_dict.items()}
    bound_indices = list(bound_dict.keys())
    preds = longitudinal_prediction[:, :, bound_indices]

    #Upper bound loss
    upper_bounds = torch.tensor([bound_dict[i][1] for i in bound_indices],device=preds.device, dtype=preds.dtype)                           
    upper_bounds_broadcast = upper_bounds.reshape((1, 1, -1))
    upper_loss = torch.relu(preds - upper_bounds_broadcast)
    return upper_loss.mean()

def domain_loss_4(longitudinal_prediction, x,option =1):
    ##TODO - Implement
    if x.is_cuda:
        device = x.get_device()
    else:
        device = torch.device("cpu")
        
    if option == 1:
        pass
    
    return torch.tensor(0.0, dtype=torch.float, device=device)

def total_loss(
    model, x, t, e, training_params,
    *,
    for_selection: bool = False,     # if True, exclude domain losses from the returned scalar
    include_rank_in_selection: bool = True,  # set False if you want NLL-only selection
    compute_parts: bool = True,       # if True, return a dict of parts for logging
    detach_domain_from_trunk: bool = False   # optional: stop domain grads hitting shared trunk
):
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
    gamma_1, gamma_2, gamma_3, gamma_4 = training_params['gamma']
    bound_dict = training_params['bound_dict']
    use_constraints = training_params.get('use_constraints', False)

    # forward
    longitudinal_prediction, outcomes = model(x)

    t, e = t.long(), e.int()

    # heads → CIFs
    cif = [torch.cumsum(ok, dim=1) for ok in outcomes]

    # core losses
    longit_loss = longitudinal_loss(longitudinal_prediction, x)
    rank_loss   = ranking_loss(cif, t, e, sigma)
    nll_loss    = negative_log_likelihood(outcomes, cif, t, e)

    # ensure non-negative combination weights
    model_loss = delta * longit_loss + alpha * rank_loss + beta * nll_loss

    # Domain losses (compute only if needed)
    domain_loss = 0.0
    dl1 = dl2 = dl3 = 0.0

    need_domain = (use_constraints and not for_selection)
    if need_domain:
        
        dl1, compliant_points, violant_points = domain_loss_1(longitudinal_prediction, x)
        dl2 = domain_loss_2(longitudinal_prediction, bound_dict)
        dl3 = domain_loss_3(longitudinal_prediction, bound_dict)
        # print(torch.isnan(violant_points[0]).all())

        # dl1_Lc = longitudinal_loss(compliant_points[1],compliant_points[0])
        # dl1_Lv = longitudinal_loss(violant_points[1],violant_points[0])
        dl1_Lc_sum, dl1_Lc_count  = clean_longitudinal(compliant_points[1],compliant_points[0])
        dl1_Lv_sum, dl1_Lv_count = clean_longitudinal(violant_points[1],violant_points[0])

        # print(dl1_Lc)
        # print(dl1_Lv)
        domain_loss = gamma_1 * dl1 + gamma_2 * dl2 + gamma_3 * dl3
    else:
        dl1, compliant_points, violant_points = domain_loss_1(longitudinal_prediction, x)
        dl1_Lc_sum, dl1_Lc_count  = clean_longitudinal(compliant_points[1],compliant_points[0])
        dl1_Lv_sum, dl1_Lv_count = clean_longitudinal(violant_points[1],violant_points[0])


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
            "dl1_Lc_sum": float(dl1_Lc_sum.cpu()),
            "dl1_Lv_sum": float(dl1_Lv_sum.cpu()),
            "dl1_Lc_count": float(dl1_Lc_count.cpu()),
            "dl1_Lv_count": float(dl1_Lv_count.cpu()),
            "domain": float(domain_loss) if need_domain else 0.0,
            "total":  float(loss_scalar),
        }

    return loss_scalar, parts

