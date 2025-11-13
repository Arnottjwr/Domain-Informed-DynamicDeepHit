'''
This file is by Vincent Jeanselme and was downloaded from:
    https://github.com/Jeanselme/DynamicDeepHit/blob/0c160824d37d6f5758e7b65d8e566aefac0901e9/ddh/losses.py
'''
import torch

def negative_log_likelihood(outcomes, cif, t, e):
    """
        Compute the log likelihood loss 
        This function is used to compute the survival loss
    """
    # assert not torch.isnan(cif[0]).any(), "NaN in cif"

    loss, censored_cif = 0, 0
    for k, ok in enumerate(outcomes):
        # Censored cif
        censored_cif += cif[k][e == 0][torch.arange((e == 0).sum()), t[e == 0]]
        # Uncensored
        selection = e == (k + 1)
        uncensored_input = ok[selection][torch.arange((selection).sum()), t[selection]] #+ 1e-10
        loss += torch.sum(torch.log(torch.clamp(uncensored_input, min=1e-10)))

    # Censored loss
        censored_input = 1 - censored_cif
    loss += torch.sum(torch.log(torch.clamp(censored_input, min=1e-10)))
    return - loss / len(outcomes)

def ranking_loss(cif, t, e, sigma):
    """
        Penalize wrong ordering of probability
        Equivalent to a C Index
        This function is used to penalize wrong ordering in the survival prediction
    """
    loss = 0
    # Data ordered by time
    for k, cifk in enumerate(cif):
        for ci, ti in zip(cifk[e-1 == k], t[e-1 == k]):
            # For all events: all patients that didn't experience event before
            # must have a lower risk for that cause
            if torch.sum(t > ti) > 0:
                loss += torch.mean(torch.exp((cifk[t > ti][torch.arange((t > ti).sum()), ti] - ci[ti])) / sigma)
    
    return loss / len(cif)

def longitudinal_loss(longitudinal_prediction, x):
    """
    Penalize error in the longitudinal predictions
    This function is used to compute the error made by the RNN

    NB: In the paper, they seem to use different losses for continuous and categorical
    But this was not reflected in the code associated (therefore we compute MSE for all)

    NB: Original paper mentions possibility of different alphas for each risk
    But take same for all (for ranking loss)
    """
    x = x[:,:,2:]  # remove age and gender
    longitudinal_prediction = longitudinal_prediction[:,:,2:]
    length = (~torch.isnan(x[:,:,0])).sum(axis = 1) - 1
    if x.is_cuda:
        device = x.get_device()
    else:
        device = torch.device("cpu")

    # Create a grid of the column index
    index = torch.arange(x.size(1)).repeat(x.size(0), 1).to(device)

    # Select all predictions until the last observed
    prediction_mask = index <= (length - 1).unsqueeze(1).repeat(1, x.size(1))
    # print(f'predict mask: {longitudinal_prediction[prediction_mask]}')

    # Select all observations that can be predicted
    observation_mask = index <= length.unsqueeze(1).repeat(1, x.size(1))
    observation_mask[:, 0] = False # Remove first observation

    X_mat = x[observation_mask]
    Y_mat = longitudinal_prediction[prediction_mask]
    valid_mask = (X_mat != -1) & (Y_mat != -1)
    X_valid = X_mat[valid_mask]
    Y_valid = Y_mat[valid_mask]
    mse_loss = torch.nn.MSELoss(reduction = 'mean')(X_valid, Y_valid)   
    return mse_loss


def domain_loss_1(longitudinal_prediction, x):
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
    return torch.nn.MSELoss(reduction = 'mean')(calculated_eGFR, predicted_eGFR_masked)   

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

def total_loss(model, x, t, e, training_params): #x: X_batch, t: Y_batch, e: D_batch

    # Compute results
    longitudinal_prediction, outcomes = model(x)
    t, e = t.long(), e.int()
    
    # Unpack parameters
    alpha = training_params['alpha']
    beta = training_params['beta']
    delta = training_params['delta']
    sigma = training_params['sigma']
    gamma_1, gamma_2, gamma_3, gamma_4 = training_params['gamma']
    bound_dict = training_params['bound_dict']
    use_constraints = training_params['use_constraints']

    # Compute cumulative function from prediced outcomes
    cif = [torch.cumsum(ok, 1) for ok in outcomes]

    longit_loss =  longitudinal_loss(longitudinal_prediction, x)
    rank_loss = ranking_loss(cif, t, e, sigma)
    nll_loss = negative_log_likelihood(outcomes, cif, t, e)
    # print(f'longit loss: {longit_loss}') 
    # print(f'rank loss  : {rank_loss}')
    # print(f'NLL loss   : {nll_loss}')

    dl1 = domain_loss_1(longitudinal_prediction,x)
    dl2 = domain_loss_2(longitudinal_prediction,bound_dict)
    dl3 = domain_loss_3(longitudinal_prediction,bound_dict)
    dl4 = domain_loss_4(longitudinal_prediction, x)
    # print(f'DL1: {dl1}')
    # print(f'DL2: {dl2}')
    # print(f'DL3: {dl3}')
    # print(f'DL4: {dl4}')
    model_loss = delta * longit_loss + alpha * rank_loss + beta * nll_loss 
    # print(float(dl1), float(dl2), float(dl3))
    if use_constraints:
        domain_loss = gamma_1 * dl1 + gamma_2 * dl2 + gamma_3 * dl3
        return model_loss + domain_loss, model_loss
    # print(nll_loss,rank_loss,longit_loss)
    return model_loss
