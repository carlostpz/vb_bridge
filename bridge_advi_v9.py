
# import packages and libraries
import torch
from torch import nn
from sklearn import datasets
import matplotlib.pyplot as plt
#from google.colab import files
import numpy as np
import sys
import random

# importing torch probability distributions
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.beta import Beta
from torch.distributions.chi2 import Chi2

#################################################################################
#                            Auxiliary functions
#################################################################################

# auxiliary function to parameterize covariate matrices by lower triangle with positive diagonal
def create_lower_tri(log_diag, lower_part, d, device):
      
  L = torch.zeros((d, d), dtype = torch.float, device = device)

  tril_indices = torch.tril_indices(row=d, col=d, offset=-1)
  L[tril_indices[0], tril_indices[1]] = lower_part
  L += torch.diag( torch.exp( log_diag ) + 10**(-6) )
  return L


# Returns a dictionary with the main_diagonal and lower triangle part of the L matrix
def extract_diag_and_lower_tri(L, device):

  d = L.shape[0]

  tril_indices = torch.tril_indices(row=d, col=d, offset=-1)
  lower_part = L[tril_indices[0], tril_indices[1]]
  main_diagonal = torch.diagonal(L)

  ans = dict(); 
  ans['main_diagonal'] = main_diagonal
  ans['lower_triangle'] = lower_part

  return ans


#######################################
#        Entropy H( q(theta) )
#######################################

# Entropy H( q(theta) )
def Entropy_Multivariate_Normal(L_log_diag):
# Args:
#
# L_log_diag: logarithm of the main diagonal in the lower triangle Choleskey decomposition of the covariate matrix
# Obs.: Notice that entropy does not depend on the mean vector for multivariate normals

  return torch.sum( L_log_diag )

#######################################
#      Log joint density p(y, theta)
#######################################

# log-likelihood
def log_lik_eval(y, X, Z, beta, gamma, phi):
    
  # average response
  mu = torch.mm(X, beta) + torch.mm(Z, gamma)
    
  # size of y vector (this will be the batch size)
  n = y.shape[0]
    
  # number of MC samples (columns of beta)
  n_mc = beta.shape[1] 
    
  # log_likelihood
  ans = torch.zeros([n_mc])
  for m in range(n_mc):
    ans[m] = -0.5*n*torch.log(2*np.pi/phi[m]) - 0.5*phi[m]*torch.sum( torch.pow(torch.squeeze(y) - mu[:,m], 2) )
    
  return ans 

# log p_gamma
def log_p_gamma_eval(gamma, sigma2_gamma):
    
  # recall the dimension of gamma
  pz = gamma.shape[0]
  n_mc = gamma.shape[1]
    
  # vector of prior density values for the distinct MC samples
  ans = torch.zeros([n_mc])
  for m in range(n_mc):
    ans[m] = -0.5*pz*np.log(2*np.pi*sigma2_gamma) - (0.5/sigma2_gamma)*torch.sum( torch.pow(gamma[:,m], 2) )
    
  return ans

# log p_beta
def log_p_beta_eval(beta, lambda_obj, alpha, phi):
  px = beta.shape[0]
  n_mc = beta.shape[1]
  ans = px*( torch.log(alpha) - np.log(2) + ( torch.log(phi) + torch.log(lambda_obj) )/alpha - torch.lgamma(1/alpha) ) 
  for m in range(n_mc):
      ans[m] = ans[m] - torch.squeeze( lambda_obj[m]*phi[m]*torch.sum( torch.abs( beta[:, m] )**alpha[m] ) )
  return ans

# log p_phi
def log_p_phi_eval(phi, a_phi, b_phi):
  p_phi_dist = Gamma(a_phi, b_phi)
  log_p_phi = p_phi_dist.log_prob(phi)
  return log_p_phi.squeeze()

# log p_lambda
def log_p_lambda_eval(lambda_obj, a_lambda, b_lambda):
  p_lambda_dist = Gamma(a_lambda, b_lambda)
  log_p_lambda = p_lambda_dist.log_prob(lambda_obj)
  return log_p_lambda.squeeze()

# log p_alpha
#def log_p_alpha_eval(alpha, a_alpha, b_alpha):
#  p_alpha_dist = Beta(a_alpha, b_alpha)
#  log_p_alpha = p_alpha_dist.log_prob(alpha/2) - np.log(2)
#  return log_p_alpha.squeeze()

def log_p_alpha_eval(alpha):
  
  p_alpha_dist_05 = Beta(100, 300)
  p_alpha_dist_1 = Beta(400, 400)
  p_alpha_dist_2 = Beta(200, 20)
  
  log_p_alpha_05 = p_alpha_dist_05.log_prob(alpha/2)/2
  log_p_alpha_1 = p_alpha_dist_1.log_prob(alpha/2)/2
  log_p_alpha_2 = p_alpha_dist_2.log_prob(alpha/2)/2
  log_p_alpha = log_p_alpha_05 + log_p_alpha_1 + log_p_alpha_2

  return log_p_alpha.squeeze()


##########################
#      Log Jacobian  
##########################

# log det(J)
def log_det_J_eval(xi_phi, xi_lambda, xi_alpha):
  return xi_phi + xi_lambda - xi_alpha - 2*torch.log( 1 + torch.exp( -xi_alpha ) )



################################################################################
################################################################################
#                       Beginning of the function
################################################################################
################################################################################

def vb_lasso_repar(X_train,
                   y_train,
                   gamma_ind,
                   output_path,
                   optim_method,
                   ols_start = True,
                   seed = 100, 
                   lambda_mean = 3.0,
                   lambda_var = 100.0,
                   phi_mean = 3.0,
                   phi_var = 100.0,
                   a_alpha = 1.0,
                   b_alpha = 1.0,
                   sigma2_gamma = 100.0,
                   n_mc = 10,
                   n_batch = 100,
                   n_epochs = 1000,
                   stop = 0.99,
                   lr = 0.001,
                   gradient_clipping = 10000,
                   save_at_epochs = 50,
                   device = torch.device("cpu")):

# ------------------------------------------------------------------------------
#   Args:
#
#   seed: random seed
#   X_train: Covariates matrix (training)
#   y_train: Response matrix (training)
#   X_test: Covariates matrix (training)
#   y_test: Response matrix (training)
#   gamma_ind: columns 0 to gamma_ind are not L_1 penalized 
#
#   lambda_mean: mean for lambda a priori (Gamma distribution)
#   lambda_var: variance for lambda a priori (Gamma distribution)
#
#   phi_mean: mean for phi a priori (Gamma distribution)
#   phi_var: variance for phi a priori (Gamma distribution)
#
#   sigma2_gamma: variance for gamma a priori ( gamma ~ N(0, sigma^2 I) )
#
#   alpha: defines and L_alpha penalization for beta
#
#   n_mc: Number of MC samples for evaluating expectations in the gradients
#   n_batch: Number of data points in the minibatch
#   n_epochs: number of eppochs during training
#   stop: stop if |ELBO_{i+1} - ELBO_i|/|ELBO_i - ELBO_{i-1}| < stop (approx. 1)
#   lr: learning rate for the gradient updates
#   ols_start: if True, start m_beta, m_gamma at unpenalized ols estimate. 
#              else, starts at 0.
#
#   optim_method: The optimization method can be
#                 "adadelta"
#                 "sgd"
#                 "adam"
#                 "adagrad"
#
#   output_path: location where to save the outputs of the function
#-------------------------------------------------------------------------------

  #n_batch = 100 


  if device == torch.device("cuda:0"):
    print("Running on GPU...")
  else:
    print("Running on CPU...")


  # Fix random seed
  torch.manual_seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
  np.random.seed(seed)  # Numpy module
  random.seed(seed)  # Python random module.
  torch.manual_seed(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True

  # setting constants to tensors
  a_alpha = torch.tensor(a_alpha, device = device)
  b_alpha = torch.tensor(b_alpha, device = device)

  # convert data from numpy to torch
  covariates = torch.from_numpy(X_train).float().to(device)

  # data dimensions
  n = list(covariates.size())[0]
  p = list(covariates.size())[1]
  
  # number of parameters
  n_param = p+3
  
  # run a random permutation on the data points so to have less variation among different batches
  permutation = np.random.choice(range(n), size=n, replace=False)
  #print(permutation)

  # permute y vector
  y = torch.from_numpy(y_train[permutation]).unsqueeze(1).float().to(device)

  # extract columns 0, ..., gama_ind from covariate matrix to form Z 
  # gamma_ind = 0
  pz = gamma_ind + 1
  Z = covariates[permutation, 0:(gamma_ind+1)].to(device)
  
  # forming covariate matrix X
  px = p - pz
  X = covariates[permutation, (gamma_ind + 1):(gamma_ind + 1 + px)].to(device)
  
  
  ###############################
  #    Prior specification
  ###############################

  # lambda ~ Ga( a_lambda, b_lambda)
  a_lambda = (lambda_mean**2)/lambda_var
  b_lambda = lambda_mean/lambda_var

  # phi ~ Ga( a_phi, b_phi)
  a_phi = (phi_mean**2)/phi_var
  b_phi = phi_mean/phi_var

  # gamma ~ N(m_gamma, sigma2_gamma * I)
  m_gamma = torch.zeros([1], device = device)
  
  # alpha ~ 2*Beta( a_alpha, b_alpha)
  
  # beta | lambda, phi, alpha ~ GG(0, (lambda*phi)^(-1/alpha) )


  #############################################
  #    Initializing variational parameters
  #############################################

  #____________________________________________________________
  # Initializing variational parameters with the OLS estimate (if ols_start = True)

  if ols_start:

    # create variational mean and Chol(covariance matrix)
    m_eps = torch.zeros([n_param, 1], dtype = torch.float, device = device)
    L_eps_log_diag = torch.zeros([n_param], dtype = torch.float, device = device)
    L_eps_lower_part = torch.zeros([int( (n_param)*(n_param-1)/2 )], dtype = torch.float, device = device)
  
    # OLS coefficients estimate
    X_full = torch.cat((Z, X), 1)
    Xt = torch.transpose(X_full, 0, 1)
    XtX = torch.mm(Xt, X_full)
    XtX_inv = torch.inverse(XtX)
    coef_ols_estimate = torch.mm( torch.mm( XtX_inv, Xt), y)

    # OLS variance estimate
    y_hat = torch.mm(X_full, coef_ols_estimate)
    sigma2_ols_estimate = torch.sum( torch.pow(y - y_hat, 2) )/(n-p)
    coef_ols_variance = XtX_inv / sigma2_ols_estimate 

    # OLS variance estimate for phi
    distrib = Chi2( torch.tensor(n-p))
    chisq_sample = distrib.sample([1000])
    mean_chisq_sample = torch.mean( torch.log(chisq_sample) )
    log_phi_ols_estimate = torch.log( torch.tensor(n-p) ) - torch.log(sigma2_ols_estimate) - mean_chisq_sample
    log_phi_ols_variance = torch.var( torch.log(chisq_sample) ).to(device)
    
    # building covariance matrix for the variational distribution
    Sigma_eps = torch.block_diag(coef_ols_variance, 
                                log_phi_ols_variance.to(device), 
                                torch.tensor(1, device = device), 
                                torch.tensor(1, device = device) )

    # extracting log diagonal and lower part of Cholesky decomposition of Sigma_eps 
    L_eps = torch.cholesky(Sigma_eps)
    diag_lower = extract_diag_and_lower_tri(L = L_eps, device = device)
    L_eps_log_diag = torch.log( diag_lower['main_diagonal'] )
    L_eps_lower_part = diag_lower['lower_triangle'] 

    # q(xi_gamma)
    m_eps[0:pz] = coef_ols_estimate[0:pz]
    
    # q(xi_beta)
    m_eps[pz:(pz+px)] = coef_ols_estimate[pz:(pz+px)]
    
    # q(xi_phi)
    m_eps[px+pz] = log_phi_ols_estimate
    
    # q(xi_lambda)
    m_eps[px+pz+1] = 0
    
    # q(xi_alpha)
    m_eps[px+pz+2] = 0

    # requiring gradients
    m_eps.requires_grad_()
    L_eps_log_diag.requires_grad_()
    L_eps_lower_part.requires_grad_()
  
  else:

    m_eps = torch.zeros([n_param, 1], dtype = torch.float, requires_grad = True, device = device)
    L_eps_log_diag = torch.zeros([n_param], dtype = torch.float, requires_grad = True, device = device)
    L_eps_lower_part = torch.zeros([int( (n_param)*(n_param-1)/2 )], requires_grad = True, dtype = torch.float, device = device)
  

  ###########################
  #     Simulate epsilon
  ###########################

  # sampling epsilon
  eps_dist = Normal(0, 1)
  epsilon = eps_dist.sample([n_param, n_mc]).float().to(device)

  ############################################
  #  List of parameters for each iteration
  ############################################

  size_output_lists = int( n_epochs / save_at_epochs )

  # q(beta) ~ N(m_beta_var, S_beta_var), S_beta_var = L_beta_var * t(L_beta_var)
  list_m_beta_var = np.zeros([px, size_output_lists])
  list_Sigma_beta_var = np.zeros([px, px, size_output_lists])

  # q(gamma) ~ N(m_gamma_var, sigma2_gamma_var * I)
  list_m_gamma_var = np.zeros([pz, size_output_lists])
  list_Sigma_gamma_var = np.zeros([pz, pz, size_output_lists])

  # q(phi) ~ log N( mu_phi_var, sigma_phi_var**2)
  list_mu_phi_var = np.zeros([size_output_lists]) 
  list_sd_phi_var = np.zeros([size_output_lists])

  # q(lambda) ~ log N( mu_lambda_var, sigma_lambda_var**2)
  list_mu_lambda_var = np.zeros([size_output_lists])
  list_sd_lambda_var = np.zeros([size_output_lists])

  # q(alpha) ~ log N( mu_alpha_var, sigma_alpha_var**2)
  list_mu_alpha_var = np.zeros([size_output_lists])
  list_sd_alpha_var = np.zeros([size_output_lists])

  # q(eps) ~ N(m_eps, Sigma_eps)
  list_m_eps = np.zeros([n_param, size_output_lists])
  list_Sigma_eps = np.zeros([n_param, n_param, size_output_lists])

  # ELBO
  list_ELBO_training = np.zeros([size_output_lists])
  list_ELBO_test = np.zeros([size_output_lists])
  

  ################################################################################
  ################################################################################
  #                                   Updates
  ################################################################################
  ################################################################################


  # torch.autograd.set_detect_anomaly(True)
  
  # Adam optimizer
  if optim_method == "adam":
    optimizer = torch.optim.Adam([m_eps, 
                                  L_eps_log_diag, 
                                  L_eps_lower_part], lr = lr )
  
  # Adadelta optimizer
  if optim_method == "adadelta":
    optimizer = torch.optim.Adadelta([m_eps, 
                                      L_eps_log_diag, 
                                      L_eps_lower_part], lr = lr )
    
  # Adagrad optimizer
  if optim_method == "adagrad":
    optimizer = torch.optim.Adagrad([m_eps, 
                                     L_eps_log_diag, 
                                     L_eps_lower_part], lr = lr )

  # SGD optimizer
  if optim_method == "sgd":
    optimizer = torch.optim.SGD([m_eps, 
                                 L_eps_log_diag, 
                                 L_eps_lower_part], lr = lr )
  
  #######################################
  #   Iterating optimization of ELBO
  #######################################

  index_save = 0
  epoch = 0
  minibatch = 0
  minibatches_per_epoch = n / n_batch
  keep_going = True

  minib_left = 0
  minib_right = n_batch

  full_ELBO = 0
  
  while keep_going:

    minib_left = minibatch * n_batch
    minib_right = minib_left + n_batch 
    
    # Extract the correspondent minibatch from X, y and Z 
    y_minib = y[minib_left:minib_right]
    X_minib = X[minib_left:minib_right, :]
    Z_minib = Z[minib_left:minib_right, :]
    
    ###################################
    #      calculate ELBO
    ###################################
    
    # create lower triangle matriz L_eps
    L_eps = create_lower_tri(log_diag = L_eps_log_diag, lower_part = L_eps_lower_part, d = n_param, device = device)
    
    # Obtain xi as a function of eps
    xi = m_eps.expand(-1, n_mc) + torch.mm(L_eps,  epsilon )
    
    # extracting entries of xi
    xi_gamma = xi[0:pz, :]
    xi_beta = xi[pz:(px+pz), :]
    xi_phi = xi[px+pz, :]
    xi_lambda = xi[px+pz+1, :]
    xi_alpha = xi[px+pz+2, :]
    
    # parameters in the original scales
    beta = xi_beta
    gamma = xi_gamma
    phi = torch.exp(xi_phi) + 10**(-6)
    lambda_obj = torch.exp(xi_lambda) + 10**(-6)
    alpha = 2/(1 + torch.exp( -xi_alpha ) )
    
    #######################
    #       Entropy q
    #######################
    
    H_xi = Entropy_Multivariate_Normal(L_eps_log_diag)
    
    #######################
    #       log p
    #######################

    # log-likelihood
    log_lik_train = log_lik_eval(y = y_minib, X = X_minib, Z = Z_minib, beta = beta, gamma = gamma, phi = phi)
    
    # log p_beta
    log_p_beta = log_p_beta_eval(beta = beta, lambda_obj = lambda_obj, alpha = alpha, phi = phi)
    
    # gamma
    log_p_gamma = log_p_gamma_eval(gamma = gamma, sigma2_gamma = sigma2_gamma)
    
    # phi
    log_p_phi = log_p_phi_eval(phi = phi, a_phi = a_phi, b_phi = b_phi)

    # lambda
    log_p_lambda = log_p_lambda_eval(lambda_obj = lambda_obj, a_lambda = a_lambda, b_lambda = b_lambda)
    
    # log_p_alpha 
    #log_p_alpha = log_p_alpha_eval(alpha = alpha, a_alpha = a_alpha, b_alpha = b_alpha)
    log_p_alpha = log_p_alpha_eval(alpha = alpha)

    # log |J|
    log_det_J = log_det_J_eval(xi_phi = xi_phi, xi_lambda = xi_lambda, xi_alpha = xi_alpha)
    
    ########################
    #       ELBO
    ########################

    likelihood_term = minibatches_per_epoch * torch.mean(log_lik_train)
    prior_term = torch.mean(log_p_gamma) + torch.mean(log_p_beta) + torch.mean(log_p_phi) + torch.mean(log_p_lambda) + torch.mean(log_p_alpha)
    log_jacobian_term = torch.mean(log_det_J)
    entropy_term = H_xi

    ELBO_minus = -1*( likelihood_term + prior_term + log_jacobian_term + entropy_term )

    ########################
    #     Optimization
    ########################
    
    # compute the gradient
    ELBO_minus.backward()
     
    # gradient clipping
    torch.nn.utils.clip_grad_norm_([m_eps, L_eps_log_diag, L_eps_lower_part], gradient_clipping)
    
    # Optimization step
    optimizer.step()
    
    # zero out the gradients
    optimizer.zero_grad()
    
    if epoch % 50 == 0:

      full_ELBO = full_ELBO - ELBO_minus.item()

      if minibatch == minibatches_per_epoch - 1:   
            
        # save current ELBO
        list_ELBO_training[index_save] = full_ELBO / minibatches_per_epoch

        # Covariance matrix for variational distribution
        Sigma_eps = torch.mm(L_eps, torch.transpose(L_eps, 0, 1) )
        
        # Save info on q(gamma) ~ N(m_gamma_var, sigma2_gamma_var * I)
        list_m_gamma_var[ :, index_save] = m_eps[0:pz, :].detach().numpy()
        list_Sigma_gamma_var[ :, :, index_save] = Sigma_eps[0:pz, 0:pz].detach().numpy()

        # Save info on q(beta) ~ N(m_beta_var, S_beta_var), S_beta_var = L_beta_var * t(L_beta_var)
        list_m_beta_var[ :, index_save] = m_eps[pz:(pz+px), :].detach().numpy()[:,0]
        list_Sigma_beta_var[ :, :, index_save] = Sigma_eps[pz:(pz+px), pz:(pz+px)].detach().numpy()

        # Save info on q(phi) ~ log N( mu_phi_var, sigma_phi_var**2)
        list_mu_phi_var[index_save] = m_eps[pz+px, :].detach().numpy()
        list_sd_phi_var[index_save] = Sigma_eps[pz+px, pz+px].detach().numpy()

        # Save info on q(lambda) ~ log N( mu_lambda_var, sigma_lambda_var**2)
        list_mu_lambda_var[index_save] = m_eps[pz+px+1, :].detach().numpy()
        list_sd_lambda_var[index_save] = Sigma_eps[pz+px+1, pz+px+1].detach().numpy()

        # Save info on q(alpha) ~ log N( mu_lambda_var, sigma_lambda_var**2)
        list_mu_alpha_var[index_save] = m_eps[pz+px+2, :].detach().numpy()
        list_sd_alpha_var[index_save] = Sigma_eps[pz+px+2, pz+px+2].detach().numpy()

        # Save info on q(eps) ~ N(m_eps, Sigma_eps) 
        list_m_eps[:, index_save] = m_eps.detach().numpy()[:,0]
        list_Sigma_eps[:, :, index_save] = Sigma_eps.detach().numpy()


        # print value of ELBOELBO_minus
        print(f'Epoch: {epoch},   Minibatch: {minibatch},   Training ELBO: { round(full_ELBO, 2) },  mean(gamma): { round(torch.mean(gamma).item(), 2)},  mean(phi): { round(torch.mean(phi).item(), 2)}, mean(lambda): { round(torch.mean(lambda_obj).item(), 2)}, mean(alpha): { round(torch.mean(alpha).item(), 2)}' )

        # empty full ELBO
        full_ELBO = 0

        # increment index_save
        index_save += 1


    if minibatch < minibatches_per_epoch - 1:
        # iterate minibatch
        minibatch += 1
    else:
        # zero out minibatch
        minibatch = 0
        # iterate epoch
        epoch += 1
        
    if (epoch == n_epochs):
        keep_going = False
    


  ##############################################################################
  #                                Save Output
  ##############################################################################

  # Saving output
  print("Saving output files...")

  # iid sampe from q(theta)
  M = 1000
  
  # sampling epsilon
  epsilon = np.random.normal( size = [n_param, M] )
  print("epsilon.shape")
  print(epsilon.shape)
  
  
  # create lower triangle matriz L_eps
  L_eps = create_lower_tri(log_diag = L_eps_log_diag, lower_part = L_eps_lower_part, d = n_param, device = device).to( torch.device("cpu") ).detach().numpy()
    
  # Obtain xi as a function of eps
  #xi = m_eps.expand(-1, M) + torch.mm(L_eps,  epsilon )
  print("m_eps.expand(-1, M).shape")
  print(m_eps.expand(-1, M).shape)
  xi = m_eps.expand(-1, M).to( torch.device("cpu") ).detach().numpy() + np.matmul(L_eps, epsilon)
    
  # extracting entries of xi
  xi_gamma = xi[0:pz, :]
  xi_beta = xi[pz:(pz+px), :]
  xi_phi = xi[px+pz, :]
  xi_lambda = xi[px+pz+1, :]
  xi_alpha = xi[px+pz+2, :]
    
  # parameters in the original scales
  beta_sample = xi_beta
  gamma_sample = xi_gamma
  phi_sample = np.exp(xi_phi) + 10**(-6)
  lambda_sample = np.exp(xi_lambda) + 10**(-6)
  alpha_sample = 2/( 1 + np.exp(-xi_alpha) )
  
  
  # saving random sample from the variational distribution
  np.savetxt( X = beta_sample, fname = output_path + "beta_sample.txt")
  np.savetxt( X = gamma_sample, fname = output_path + "gamma_sample.txt")
  np.savetxt( X = phi_sample, fname = output_path + "phi_sample.txt")
  np.savetxt( X = lambda_sample, fname = output_path + "lambda_sample.txt")
  np.savetxt( X = alpha_sample, fname = output_path + "alpha_sample.txt")

  # saving the variational parameters
  np.savetxt( X = list_m_gamma_var, fname = output_path + "list_m_gamma.txt")
  np.savetxt( X = list_m_beta_var, fname = output_path + "list_m_beta.txt")
  np.savetxt( X = list_mu_phi_var, fname = output_path + "list_m_phi.txt")
  np.savetxt( X = list_mu_lambda_var, fname = output_path + "list_m_lambda.txt")
  np.savetxt( X = list_mu_alpha_var, fname = output_path + "list_m_alpha.txt")
  
  np.save( arr = list_Sigma_gamma_var, file = output_path + "list_Sigma_gamma.npy")
  np.save( arr = list_Sigma_beta_var, file = output_path + "list_Sigma_beta.npy")
  np.savetxt( X = list_sd_phi_var, fname = output_path + "list_sd_phi.txt")
  np.savetxt( X = list_sd_lambda_var, fname = output_path + "list_sd_lambda.txt")
  np.savetxt( X = list_sd_alpha_var, fname = output_path + "list_sd_alpha.txt")

  np.savetxt( X = list_m_eps, fname = output_path + "list_m_eps.txt")
  np.save( arr = list_Sigma_eps, file = output_path + "list_Sigma_eps.npy")

  # ELBO
  np.savetxt( X = list_ELBO_training, fname = output_path + "ELBO_array_training.txt")
  np.savetxt( X = list_ELBO_test, fname = output_path + "ELBO_array_test.txt")

  # Output saved
  print("Output files saved in " + output_path)

  # End
  print("Done.")




