
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

#######################################
#      Log joint density p(y, theta)
#######################################

# log-likelihood
def log_lik_eval(y, X, Z, beta, gamma, phi):
    
  # average response
  mu = torch.mm(X, beta) + torch.mm(Z, gamma)
    
  # size of y vector (this will be the batch size)
  n = y.shape[0]
    
  # log_likelihood
  ans = 0.5*n*torch.log(phi) - 0.5*phi*torch.sum( torch.pow(torch.squeeze(y) - torch.squeeze(mu), 2) )
    
  return ans 

# log p_gamma
def log_p_gamma_eval(gamma, sigma2_gamma):
    
  # recall the dimension of gamma
  pz = gamma.shape[0]
    
  # prior density for gamma
  ans = - 1/(2*sigma2_gamma)*torch.sum( torch.pow(gamma, 2) )
    
  return ans

# log p_beta
def log_p_beta_eval(beta, lambda_obj, alpha, phi):
  px = beta.shape[0]
  ans = px*( torch.log(alpha) + 0.5*torch.log(phi) + torch.log(lambda_obj)/alpha - torch.lgamma(1/alpha) ) 
  ans = ans - lambda_obj*torch.pow(phi, alpha/2)*torch.squeeze( torch.sum( torch.pow( torch.abs( beta ), alpha ) ) )
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
def log_p_alpha_eval(alpha, a_alpha, b_alpha):
  p_alpha_dist = Beta(a_alpha, b_alpha)
  log_p_alpha = p_alpha_dist.log_prob(alpha/2.5) - np.log(2.5)
  return log_p_alpha.squeeze()

#def log_p_alpha_eval(alpha):
#  
#  p_alpha_dist_05 = Beta(100, 300)
#  p_alpha_dist_1 = Beta(400, 400)
#  p_alpha_dist_2 = Beta(200, 20)
#  
#  p_alpha_05 = torch.exp( p_alpha_dist_05.log_prob(alpha/2.5) ) / 2.5
#  p_alpha_1 = torch.exp( p_alpha_dist_1.log_prob(alpha/2.5) ) / 2.5
#  p_alpha_2 = torch.exp( p_alpha_dist_2.log_prob(alpha/2.5) ) / 2.5
#  log_p_alpha = torch.log( ( p_alpha_05 + p_alpha_1 + p_alpha_2 ) / 3 )
#
#  return log_p_alpha.squeeze()



################################################################################
################################################################################
#                       Beginning of the function
################################################################################
################################################################################

def bridge_map(X_train,
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
               lambda_init = 5,
               alpha_init = 1,
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


  ##################################
  #    Initializing parameters
  ##################################

  #____________________________________________________________
  # Initializing variational parameters with the OLS estimate (if ols_start = True)

  if ols_start:

    # OLS coefficients estimate
    X_full = torch.cat((Z, X), 1)
    Xt = torch.transpose(X_full, 0, 1)
    XtX = torch.mm(Xt, X_full)
    XtX_inv = torch.inverse(XtX)
    coef_ols_estimate = torch.mm( torch.mm( XtX_inv, Xt), y)

    # OLS variance estimate
    y_hat = torch.mm(X_full, coef_ols_estimate)
    sigma2_ols_estimate = torch.sum( torch.pow(y - y_hat, 2) )/(n-p)
    
    # initializing variables
    gamma_var = coef_ols_estimate[0:pz, ].float().to(device=device).requires_grad_()
    beta_var = coef_ols_estimate[pz:(pz+px), ].float().to(device=device).requires_grad_()
    log_phi_var = torch.tensor([ torch.log(1 / sigma2_ols_estimate) ], dtype = torch.float, requires_grad = True, device = device)
    log_lambda_var = torch.tensor([np.log(lambda_init)], dtype = torch.float, requires_grad = True, device = device)
    real_alpha_var = torch.tensor([ np.log(alpha_init) - np.log(2 - alpha_init) ], dtype = torch.float, requires_grad = True, device = device)

  else:

    gamma_var = torch.zeros([pz, 1], dtype = torch.float, requires_grad = True, device = device)
    beta_var = torch.zeros([px, 1], dtype = torch.float, requires_grad = True, device = device)
    log_phi_var = torch.zeros([1], dtype = torch.float, requires_grad = True, device = device)
    log_lambda_var = torch.zeros([1], dtype = torch.float, requires_grad = True, device = device)
    real_alpha_var = torch.zeros([1], dtype = torch.float, requires_grad = True, device = device)
    

  size_output_lists = int( n_epochs / save_at_epochs )

  # q(beta) ~ N(m_beta_var, S_beta_var), S_beta_var = L_beta_var * t(L_beta_var)
  list_beta = np.zeros([px, size_output_lists])
  list_gamma = np.zeros([pz, size_output_lists])
  list_phi = np.zeros([size_output_lists]) 
  list_lambda = np.zeros([size_output_lists])
  list_alpha = np.zeros([size_output_lists])

  # joint distribution
  list_log_joint_training = np.zeros([size_output_lists])
  list_log_joint_test = np.zeros([size_output_lists])
  

  ################################################################################
  ################################################################################
  #                                   Updates
  ################################################################################
  ################################################################################


  # torch.autograd.set_detect_anomaly(True)
  
  # Adam optimizer
  if optim_method == "adam":
    optimizer = torch.optim.Adam([beta_var, 
                                  gamma_var, 
                                  log_phi_var, 
                                  log_lambda_var,
                                  real_alpha_var], 
                                  lr = lr )
  
  # Adadelta optimizer
  if optim_method == "adadelta":
    optimizer = torch.optim.Adadelta([beta_var, 
                                      gamma_var, 
                                      log_phi_var, 
                                      log_lambda_var,
                                      real_alpha_var], lr = lr )
    
  # Adagrad optimizer
  if optim_method == "adagrad":
    optimizer = torch.optim.Adagrad([beta_var, 
                                    gamma_var, 
                                    log_phi_var, 
                                    log_lambda_var,
                                    real_alpha_var], lr = lr )

  # SGD optimizer
  if optim_method == "sgd":
    optimizer = torch.optim.SGD([beta_var, 
                                gamma_var, 
                                log_phi_var, 
                                log_lambda_var,
                                real_alpha_var], lr = lr )
  
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
  
  full_log_joint = 0

  while keep_going:

    minib_left = minibatch * n_batch
    minib_right = minib_left + n_batch 
    
    # Extract the correspondent minibatch from X, y and Z 
    y_minib = y[minib_left:minib_right]
    X_minib = X[minib_left:minib_right, :]
    Z_minib = Z[minib_left:minib_right, :]
    
    # calculating parameters in their original scale
    phi_var = torch.exp(log_phi_var) + 10**(-6)
    lambda_var = torch.exp(log_lambda_var) + 10**(-6)
    alpha_var = 2/(1 + torch.exp(-real_alpha_var) )

    ###################################
    #      calculate log-likelihood
    ###################################
    

    # log-likelihood
    log_lik_train = minibatches_per_epoch * log_lik_eval(y = y_minib, X = X_minib, Z = Z_minib, beta = beta_var, gamma = gamma_var, phi = phi_var)
    
    #######################
    #       log p
    #######################

    # log p_beta
    log_p_beta = log_p_beta_eval(beta = beta_var, lambda_obj = lambda_var, alpha = alpha_var, phi = phi_var)
    
    # gamma
    log_p_gamma = log_p_gamma_eval(gamma = gamma_var, sigma2_gamma = sigma2_gamma)
    
    # phi
    log_p_phi = log_p_phi_eval(phi = phi_var, a_phi = a_phi, b_phi = b_phi)

    # lambda
    log_p_lambda = log_p_lambda_eval(lambda_obj = lambda_var, a_lambda = a_lambda, b_lambda = b_lambda)
    
    # log_p_alpha 
    log_p_alpha = log_p_alpha_eval(alpha = alpha_var, a_alpha = a_alpha, b_alpha = b_alpha)
    #log_p_alpha = log_p_alpha_eval(alpha = alpha_var)


    ########################
    #     log-joint
    ########################

    # log_joint density
    log_joint_minus = -1*( log_lik_train + log_p_gamma + log_p_beta + log_p_phi + log_p_lambda + log_p_alpha )

    ########################
    #     Optimization
    ########################
    
    # compute the gradient
    log_joint_minus.backward()
     
    # gradient clipping
    torch.nn.utils.clip_grad_norm_([beta_var, gamma_var, log_phi_var, log_lambda_var, real_alpha_var], gradient_clipping)
    
    # Optimization step
    optimizer.step()
    
    # zero out the gradients
    optimizer.zero_grad()
    
    if epoch % 50 == 0:

      # every 50 epochs, we calculate the full log joint distribution by accumulating
      # it over all data batches
      full_log_joint = full_log_joint - log_joint_minus.item()

      if minibatch == minibatches_per_epoch - 1:   
            
        # save current MAP
        list_log_joint_training[index_save] = - full_log_joint / minibatches_per_epoch
        
        # Save info on gamma
        list_gamma[ :, index_save] = gamma_var.squeeze(1).detach().numpy()

        # Save info on q(beta) ~ N(m_beta_var, S_beta_var), S_beta_var = L_beta_var * t(L_beta_var)
        list_beta[ :, index_save] = beta_var.squeeze(1).detach().numpy()

        # Save info on q(phi) ~ log N( mu_phi_var, sigma_phi_var**2)
        list_phi[index_save] = phi_var.detach().numpy()

        # Save info on q(lambda) ~ log N( mu_lambda_var, sigma_lambda_var**2)
        list_lambda[index_save] = lambda_var.detach().numpy()

        # Save info on q(alpha) ~ log N( mu_lambda_var, sigma_lambda_var**2)
        list_alpha[index_save] = alpha_var.detach().numpy()

        # print value of the objective function
        print(f'Epoch: {epoch},   Minibatch: {minibatch},   Training objective: { round(full_log_joint, 2) },  phi: { round(phi_var.item(), 5)}, lambda: { round(lambda_var.item(), 2)}, alpha: { round(alpha_var.item(), 2)}' )

        # empty full log joint density
        full_log_joint = 0

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
        
    # trining must stop when epoch reaches n_epochs
    if (epoch == n_epochs):
        keep_going = False
    


  ##############################################################################
  #                                Save Output
  ##############################################################################

  # Saving output
  print("Saving output files...")

  # saving the variational parameters
  np.savetxt( X = list_gamma, fname = output_path + "list_gamma.txt")
  np.savetxt( X = list_beta, fname = output_path + "list_beta.txt")
  np.savetxt( X = list_phi, fname = output_path + "list_phi.txt")
  np.savetxt( X = list_lambda, fname = output_path + "list_lambda.txt")
  np.savetxt( X = list_alpha, fname = output_path + "list_alpha.txt")
  
  # ELBO
  np.savetxt( X = list_log_joint_training, fname = output_path + "log_joint_training.txt")
  np.savetxt( X = list_log_joint_test, fname = output_path + "log_joint_test.txt")

  # Output saved
  print("Output files saved in " + output_path)

  # End
  print("Done.")




