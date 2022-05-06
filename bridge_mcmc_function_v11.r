
library(tictoc)
library(mvtnorm)

#################################################################
#################################################################
#################################################################
#################################################################
###############         MCMC function         ###################
#################################################################
#################################################################
#################################################################
#################################################################


mcmc_bayesian_bridge = function(y, 
                                covariates, 
                                gamma_ind, 
                                alpha = 1, 
                                lambda = 1,
                                n_iter, 
                                a_lambda = 1, 
                                b_lambda =1,
                                a_phi = 1, 
                                b_phi = 1,
                                a_eta = 1,
                                b_eta = 1,
                                sample_alpha = FALSE,
                                sample_lambda = TRUE,
                                V = 1,
                                n_iter_em = 1000,
                                burnin,
                                seed = 100){
  
  # Args
  #_____________________________________________________________________
  #
  # y: Responses (numeric vector)
  #  
  # covariates: Covariates (numeric matrix)
  #  
  # gamma_ind: Column indexes that form the matrix Z (numeric vector).
  #            If gamma_ind = 0, then there is no matrix Z (no unpenalized regression coefficients) 
  #
  # alpha: If sample_alpha == FALSE, alpha will be fixed at this value throughout the mcmc
  #        If sample_alpha == TRUE, alpha will be initialized at this value
  #         (numeric)
  #
  # lambda: If sample_lambda == FALSE, alpha will be fixed at this value throughout the
  #         mcmc. If sample_lambda == TRUE, alpha will be initialized at this value
  #         (numeric)
  #
  # covariates_contains_intercept: Is the intercept already inside the covariates 
  #                                matrix? This function ALWAYS include the intercept 
  #                                as a parameter to be estimated. If covariates do not
  #                                include intercept, this functions adds it. If cova-
  #                                already contains an intercept, this function leaves
  #                                it as it is.
  #                                (boolean)
  #
  # n_iter: number of mcmc iterations (numeric)
  #
  # sample_lambda: Should alpha be estimated? (boolean)
  #
  # sample_alpha: Should alpha be treated as a parameter to be estimated? (boolean)
  #               If FALSE, alpha is fixed at the argument alpha passed to the function
  #  
  # V: variance of the random walk MH proposal. Ignored if sample_alpha is FALSE
  #
  #
  #   Default prior specification
  #_____________________________________________________________________
  #
  #  1) lambda ~Ga(a_lambda, b_lambda)
  #                 a_lambda = 1; b_lambda = 1
  #  
  #  2) beta_j ~ Unif(- u_j^{1/alpha}, u_j^{1/alpha})
  #  
  #  3) phi ~ Gamma(a_phi, b_phi) 
  #                 a_phi = 1; b_phi = 1
  #  
  #  4) u_j ~ Gamma(1/alpha + 1, lambda) 
  #  
  #  5) gamma ~ N(mu_gamma, Sigma_gamma)
  #               mu_gamma = rep(0, pz) 
  #               Sigma_gamma = diag(pz)
  #
  #  6) alpha = 2.5*eta,  eta ~ beta(a_eta, b_eta)
  #  
  
  # returns: 
  #_____________________________________________________________________
  #
  # list with mcmc chains for each parameter. Burn-in needs to be done afterwards
  
  
  ##############################################
  ########    Auxiliary functions     ##########
  ##############################################
  
  
  #   truncated univariate normal as in Damien and Walker (2001)
  #################################################################
  my_truncated_gaussian = function(x,
                                   mu,
                                   sigma2,
                                   lower,
                                   upper){
    
    # routine that samples from X ~ N(mu, sigma^2)I(a < X < b) 
    # via Gibbs sampler with augmented variables
    
    # sampling auxiliary variable
    y_aux = runif(1, min = 0, max = exp(- (0.5/sigma2) * (x-mu)^2 ) )
    
    # sampling X ~ N(mu, sigma^2)I(a < X < b)
    x = runif(1, 
              min = max(lower, mu - sqrt(-2*sigma2 * log(y_aux)) ), 
              max = min(upper, mu + sqrt(-2*sigma2 * log(y_aux)) ))
    
    return(x)
    
  }
  
  
  # truncated multivariate normal as in Damien and Walker (2001)
  #################################################################
  my_tmvn = function(x,
                     mu,
                     Sigma_inv,
                     lower,
                     upper){
    
    p = length(mu)
    S = Sigma_inv
    left_trunc = rep(0, p)
    right_trunc = rep(0, p)
    
    # sample auxiliary variable
    y_star = rexp(1, rate = 0.5)
    y_aux = y_star + as.numeric( t(x-mu)%*%Sigma_inv%*%(x-mu) )
    
    # sample x
    for(k in 1:p){
      
      # truncating region
      ak = S[k,k]
      bk = 2 * sum( S[k, -k]*(x[-k] - mu[-k]) )
      ck_matrix = (x[-k] - mu[-k]) %*% t( x[-k] - mu[-k] ) * S[-k, -k]
      ck = sum( ck_matrix ) - y_aux
      
      ellipse_left = mu[k] -bk/(2*ak) - sqrt(bk^2-4*ak*ck) / (2*abs(ak) )
      ellipse_right = mu[k] -bk/(2*ak) + sqrt(bk^2-4*ak*ck) / (2*abs(ak) )
      
      left_trunc[k] = max( ellipse_left, lower[k])
      right_trunc[k]= min( ellipse_right, upper[k])
      
      x[k] = runif(1, min = left_trunc[k], max = right_trunc[k] )
      
    }
    
    return( x )
    
  }
  
  
  # sample from truncated inverse gamma as in Damien and Walker (2001)
  #######################################################################
  trunc_gamma_inv_gibbs = function(x, alpha, beta, a=0, b=Inf){
    
    # p(y|x)
    log_y = -beta/x + log(runif(1))
    
    # p(x|y)
    my = max(a, -beta/log_y )
    U = runif(1)
    result = my / ( ( U * ( (my/b)^alpha - 1 ) + 1 )^(1/alpha) )
    
    return(result)
    
  }
  
  # sample from truncated gamma as in Damien and Walker (2001)
  ###################################################################
  trunc_gamma_gibbs = function(x, shape, rate, a=0, b=Inf){
    
    # p(y|x)
    log_y = -rate*x + log(runif(1))
    
    # p(x|y)
    M_y_b_rate = min(b, -log_y/rate )
    U = runif(1)
    if( a == 0 ){
      result = ( U^(1/shape) ) * M_y_b_rate
    }else{
      result = ( U * ( M_y_b_rate^shape - a^shape ) + a^shape )^(1/shape)
    }
    
    return(result)
    
  }
  
  
  # Evaluates alpha domain
  ##########################
  alpha_domain = function(phi, beta, u){
    
    #print( abs(beta) )
    if ( sum( abs(beta) > 1/sqrt(phi) ) == 0 ){
      
      # if there are no beta_j greater than 1 (S_plus = empty), then upper limit equals 2
      upper_limit = 2.5
      
    }else{ 
      
      # in case S_plus is non-empty,
      S_plus = which( abs(beta) > 1/sqrt(phi) )
      upper_limit = min( 2.5, (log(u[S_plus]) / ( log( abs(beta[S_plus]) ) + 0.5*log(phi) ) ) )
    }
    
    if( sum( abs(beta) < 1/sqrt(phi) & abs(beta) > 0 ) == 0 ){
      
      # if there are no beta_j in (0, 1), ie S_minus = empty, the lower limit equals 0
      lower_limit = 0
    }else{ 
      
      # in case S_minus is non-empty,
      S_minus = intersect( which( abs(beta) < 1/sqrt(phi) ), which( abs(beta) > 0 ) )
      lower_limit = max( 0, log(u[S_minus]) / ( log( abs(beta[S_minus]) ) + 0.5 * log(phi) ) )
      
    }
    
    return( c(lower_limit, upper_limit) )
    
  }
  
  # Evaluates log joint distribution
  ####################################

  # log likelihood
  log_lik_fun = function(y, X, Z, beta, gamma, phi){
    ans = sum( dnorm( as.numeric(y), mean = as.numeric(X%*%beta + Z%*%gamma), sd = 1/sqrt(phi), log = TRUE ) )
    return( ans )
  }
  
  # log p(phi)
  log_phi_fun = function(phi, a_phi, b_phi){
    ans = dgamma(phi, shape = a_phi, rate = b_phi, log = TRUE)
    return( ans )
  }

  # log p(lambda)
  log_lambda_fun = function(lambda, a_lambda, b_lambda){
    ans = dgamma( lambda, shape = a_lambda, rate = b_lambda, log = TRUE)
    return( ans )
  }

  # log p(gamma)
  log_gamma_fun = function(gamma, mu_gamma, Sigma_gamma){
    ans = dmvnorm( gamma, mean = mu_gamma, sigma = Sigma_gamma, log = TRUE)
    return( ans )
  }

  # log p(beta)
  log_beta_fun = function(beta, lambda, phi, alpha){
    px = length( as.numeric(beta) )
    ans = px*( log(alpha) + log(lambda)/alpha + log(phi)/2 - lgamma(1/alpha) ) - lambda*phi^(alpha/2) * sum( abs(beta)^alpha )
    return( ans )
  }

  # log p(alpha)
  log_alpha_fun = function(alpha, a_eta, b_eta){
    ans = dbeta(alpha/2.5, shape1 = a_eta, shape2 = b_eta, log = TRUE) - log(2.5)
    return( ans )
  }

  # log joint distribution: log p(y, theta)
  log_joint_fun = function(y, X, Z, beta, gamma, phi, lambda, alpha,
                           a_phi, b_phi, a_lambda, b_lambda, mu_gamma, Sigma_gamma, a_eta, b_eta ){

    ans = 0
    ans = ans + log_lik_fun(y=y, X=X, Z=Z, beta = beta, gamma = gamma, phi = phi)
    ans = ans + log_phi_fun(phi = phi, a_phi = a_phi, b_phi = b_phi)
    ans = ans + log_lambda_fun(lambda = lambda, a_lambda = a_lambda, b_lambda = b_lambda)
    if( length(gamma_ind) > 0 ){
      ans = ans + log_gamma_fun(gamma = gamma, mu_gamma = mu_gamma, Sigma_gamma = Sigma_gamma)  
    }
    ans = ans + log_beta_fun(beta = beta, lambda = lambda, phi = phi, alpha = alpha)
    ans = ans + log_alpha_fun( alpha = alpha, a_eta = a_eta, b_eta = b_eta)

  }


  #_________________________________________________________________________________
  
  # set random seed (default = 100)
  set.seed(seed)
  
  ##############################################
  ##############################################
  ########            MCMC            ##########
  ##############################################
  ##############################################
  
  
  #####################
  #   Initialization
  #####################
  
  
  p = ncol(covariates) # This COUNTS the intercept
  n = nrow(covariates)
  
  if (length(gamma_ind) > 0){
    Z = covariates[, gamma_ind]
    if ( length(gamma_ind)==1 ){
      Z = matrix(Z, ncol = 1)
    }
    X = covariates[, setdiff(1:p, gamma_ind)]
  }else{
    X = covariates
    Z = matrix( rep(1, n), ncol = 1)  # Z can be arbitrary because gamma_par will be stuck at 0
  }
  pz = length(gamma_ind)
  px = p - pz 
  
  XtX = t(X)%*%X
  # it only makes sense to invert t(X)X if n > px
  if( n > px ){
    XtX_inv = solve( t(X)%*%X )
  }
  
  # ZtZ and ZtZ_inv could be anything if length(gamma_ind) > 0 because gamma will be always 0
  ZtZ = t(Z)%*%Z
  ZtZ_inv = solve( t(Z)%*%Z )
  
  if ( n > px ){
    beta_ols = as.numeric( solve(t(covariates)%*%covariates + 0.00001*diag(px+pz) )%*%t(covariates)%*%y )
    if( length(gamma_ind) > 0 ){
      beta = beta_ols[-gamma_ind]
      gamma_par = beta_ols[gamma_ind] # the name gamma_par is here to distinguish from the R function gamma()
    }else{
      beta=beta_ols
      gamma_par = 0 # gamma must be zero trhoughout the MCMC 
    }
  }else{
    beta = rep(0, px)
    gamma_par = rep( mean(y), pz )
  }
  #phi = 4
  #phi = 1/ mean( (y - X%*%beta - Z%*%gamma_par)^2 )
  phi = 1
  u = phi^(alpha/2)*( abs(beta)^alpha ) + 1
  
  if( length(gamma_ind) > 0 ){
    mu_gamma = rep(0, length(gamma_ind) ) 
    Sigma_gamma = diag( length(gamma_ind) )
    Sigma_gamma_inv = solve(Sigma_gamma)
  }else{
    mu_gamma = 999999 # these are arbitrary and should not affect the mcmc
    Sigma_gamma = 999999  # these are arbitrary and should not affect the mcmc
  }

  # the initial values for lambda and alpha are specified by the user 
  # (default = 1 for both)
  
  # Initializing alpha:
  #
  # First, check if user specified initial value for alpha is within its support
  # If it is not, tell the user and initialize alpha at the center of its support
  #alpha_bounds = alpha_domain(phi=phi, beta=beta, u=u)
  #if ( alpha < alpha_bounds[1] | alpha > alpha_bounds[2] ){
  #  alpha = mean(alpha_bounds)
  #}

  # create mcmc chain
  chain_beta = matrix(0, ncol = n_iter, nrow = px)
  chain_gamma = matrix(0, ncol = n_iter, nrow = pz)
  chain_u = matrix(0, ncol = n_iter, nrow = px)
  chain_phi = rep(0, n_iter)
  chain_lambda = rep(0, n_iter)
  chain_alpha = rep(0, n_iter)
  accept_prob = 0
  chain_log_joint = rep(0, n_iter)
  iteration_times = rep(0, n_iter)
  cummulative_times = rep(0, n_iter)
  
  # records hyperparameters used in the prior specification
  hyperparameters = list( "a_lambda" = a_lambda,
                          "b_lambda" = b_lambda,
                          "a_phi" = a_phi,
                          "b_phi" = b_phi,
                          "mu_gamma" = mu_gamma,
                          "Sigma_gamma" = Sigma_gamma,
                          "V" = V )

  for (iter in 1:n_iter){
    
    tic()
    
    if (iter %% 10 == 0){
      print(iter)
    }
    

    ########################
    #   sampling alpha
    ########################
      
    if (sample_alpha){
        
      v_new = rnorm(n=1,
                    mean = log(alpha) - log(2.5-alpha),
                    sd = sqrt(V) )
      
      alpha_new = 2.5/( 1 + exp(-v_new) )
      
      log_ratio = px*( log(alpha_new) - log(alpha) ) + 
          + px * ( lgamma(1/alpha) - lgamma(1/alpha_new) ) +
          + a_eta * log(alpha_new) + b_eta * log(2.5 - alpha_new) +
          - a_eta * log(alpha) - b_eta * log(2.5 - alpha) +
          - lambda * sum( abs(beta)^alpha_new * phi^(alpha_new/2) - abs(beta)^alpha * phi^(alpha/2) ) +
          + (px/alpha_new - px/alpha)* log(lambda)
        
      coin = log( runif(1) )
        
      # accept alpha_new
      if( coin < log_ratio ){ 

        # update alpha
        alpha = alpha_new 
          
        # acceptance probability
        accept_prob = accept_prob + 1  
      }
        
    }
    
    chain_alpha[iter] = alpha


    ################
    #  sampling u
    ################
    
    u_star = rep(0, px)
    for(k in 1:px){
      u_star[k] = rexp(n=1, rate = lambda)
    }
    u = u_star + ( phi^(alpha/2) )*( abs(beta)^alpha )
    chain_u[, iter] = u
    

    ##################
    #  sampling phi
    ##################
    
    phi = trunc_gamma_gibbs(x = phi,
                            shape = n/2 + px/2 + a_phi,
                            rate = 0.5*sum( (y - X %*% beta - Z%*%gamma_par)^2 ) + b_phi,
                            a=0, 
                            b = min( u^(2/alpha)*( beta^(-2) ) ) )
      
    #phi = 1/trunc_gamma_inv_gibbs(x = phi, 
    #                              alpha = a_phi + n/2 + px/alpha, 
    #                              beta =  0.5*sum( (y - X %*% beta - Z%*%gamma_par)^2 ) + b_phi, 
    #                              a=max( 1/u*( abs(beta)^alpha ) ), 
    #                              b=Inf)
    
    chain_phi[iter] = phi
    

    ################
    # sampling beta
    ################
    
    if(px > n){
      
      # vector that holds the truncated gaussian means
      mu = rep(0, px)
      
      # if px > n we cannot invert XtX, so we sample each beta_j separately
      for (j in 1:px){
        obj = ( as.numeric( XtX[j, -j]%*%beta[-j] ) - t(y-Z%*%gamma_par) %*% X[ ,j] )/ sum( X[,j]^2 )
        mu[j] = as.numeric(obj)
        beta[j] = my_truncated_gaussian(x = beta[j],
                                        mu = mu[j],
                                        sigma2 = 1/( phi*sum( X[,j]^2 ) ),
                                        lower = -( u[j]^(1/alpha) )/sqrt(phi),
                                        upper = ( u[j]^(1/alpha) )/sqrt(phi) )
      }
      
    }else{
      beta = my_tmvn( x = beta,
                      mu = as.numeric( XtX_inv%*%t(X)%*%(y - Z%*%gamma_par) ),
                      Sigma_inv = XtX * phi,
                      lower = -( u^(1/alpha) )/sqrt(phi),
                      upper =  ( u^(1/alpha) )/sqrt(phi) )
    }
    
    chain_beta[, iter] = beta
    
    ##################
    # sampling gamma
    ##################
    
    # we only sample gamma if length(gamma_ind)>0, i.e., if there is at least one unpenalized regression
    # coefficient
    if (length(gamma_ind) > 0){
      var_gamma = solve( ZtZ*phi +  Sigma_gamma_inv )
      gamma_par = as.numeric( rmvnorm(n=1, 
                          mean = var_gamma%*%(phi*t(Z)%*%(y - X%*%beta) + Sigma_gamma_inv%*%mu_gamma),
                          sigma = var_gamma ) )
      chain_gamma[, iter] = gamma_par
    }
    
    ########################
    #   sampling lambda
    ########################
    
    if(sample_lambda){
      lambda = rgamma(n = 1, 
                      shape = a_lambda + px/alpha, 
                      rate = b_lambda + phi^(alpha/2)*sum( abs(beta)^alpha ) )
    }
    chain_lambda[iter] = lambda
    

    #########################
    #   evaluate log-joint
    #########################
    chain_log_joint[iter] = log_joint_fun( y = y, X = X, Z = Z, beta = beta, gamma = gamma_par, phi = phi, lambda = lambda, alpha = alpha,
                                           a_phi = a_phi, b_phi = b_phi, a_lambda = a_lambda, b_lambda = b_lambda, 
                                           mu_gamma = mu_gamma, Sigma_gamma = Sigma_gamma, a_eta = a_eta, b_eta = b_eta )


    #########################
    #      record time
    #########################
    time_list = toc( quiet = TRUE )
    iteration_times[iter] = time_list$toc - time_list$tic
    
  }
  
  # accumulate the times spent at each iteration
  cummulative_times = cumsum(iteration_times)
  
  ########################################
  ########################################
  #######    MCMC point estimates  #######
  ########################################
  ########################################

  mcmc_index = ( burnin + 1 ):n_iter
  
  beta_hat = apply(X = chain_beta[, mcmc_index], FUN = mean, MARGIN = 1)
  if (nrow(chain_gamma) == 1){
    gamma_hat = mean( chain_gamma[, mcmc_index] )
  }else{
    gamma_hat = apply(X = chain_gamma[, mcmc_index], FUN = mean, MARGIN = 1)
  }
  lambda_hat = mean(chain_lambda[mcmc_index])
  phi_hat = mean(chain_phi[mcmc_index])
  alpha_hat = mean(chain_alpha[mcmc_index])
 
  ########################################
  ########################################
  #######     MAP given lambda     #######
  ########################################
  ########################################
  
  #lambda_hat = mean( chain_lambda[mcmc_index] )
  #glmnet_obj = glmnet(x=X, y=y, family = "gaussian", alpha = 1, 
  #                    standardize = FALSE, lambda = lambda_hat )
  #beta_hat_lasso_fixed_lambda = as.numeric( glmnet_obj$beta )
  #gamma_hat_lasso_fixed_lambda = as.numeric( glmnet_obj$a0 )
  
  ########################################
  ########################################
  #######     EM Algorithm     ###########
  ########################################
  ########################################
  

  # initializing EM beta chain
  EM_beta_chain = matrix(0, ncol = n_iter_em, nrow = px)
   
  '
  a_lambda = hyperparameters$a_lambda
  b_lambda = hyperparameters$b_lambda
  
  # initializing beta
  #beta = beta_hat_ols[-1]
  beta = apply(X = chain_beta, MARGIN = 1, FUN = mean)
  #beta = rep(0, px)
  
  # initializing beta_new and lambda_star
  beta_new = rep(0, px)
  lambda_star = rep(0, px)
  
  # auxiliary quantities (fixed throughout the algorithm)
  #Z = matrix( rep(1, n), ncol = 1 )
  z = y - Z%*%gamma_hat

  # if n > p then beta_ols is available to initialize the algorithm
  if (n > p) {
    XtX = t(X)%*%X
    XtX_inv = solve(XtX)
    beta_hat_0 = XtX_inv%*%t(X)%*%z
  }else{ 
    # in case n < p we have to initialize the algorithm differently
    beta_hat_0 = apply(X = chain_beta, MARGIN = 1, FUN = mean)
  }
  
  
  for (iter in 1:n_iter_em){
    
    # calculating lambda_hat
    lambda_hat = (px/alpha_hat + a_lambda) / (b_lambda + phi_hat*sum( abs(beta)^alpha_hat ) )
    
    for (j in 1:px ){
      
      # calculating lambda_star
      lambda_star[j] = alpha_hat * lambda_hat * ( abs( beta_hat_0[j] )^(alpha_hat - 1) )
      
      # Updating beta
      rj = y - Z %*% gamma_hat - X[, -j]%*%beta[-j]
      beta_hat_ols_j = X[, j]%*%rj / sum( X[,j]^2 )
      threshold_j = lambda_star[j] / sum( X[,j]^2 )
      
      beta_new[j] = (beta_hat_ols_j - threshold_j) * I( beta_hat_ols_j > threshold_j) + 
        + (beta_hat_ols_j + threshold_j) * I( beta_hat_ols_j < -threshold_j)
      
      beta[j] = beta_new[j]
    }
    
    #beta = beta_new
    EM_beta_chain[, iter] = beta
    
  }
  '
  
  # registering total time elapsed
  time = cummulative_times[n_iter]
  print(time)
  
  # list of mcmc chains and prior specification to be returned by the function
  if (sample_alpha){
    
    accept_prob = accept_prob/n_iter
    
    ans = list("beta" = chain_beta,
               "gamma" = chain_gamma,
               "phi" = chain_phi,
               "lambda" = chain_lambda,
               "u" = chain_u,
               "alpha" = chain_alpha,
               "sample_alpha" = sample_alpha,
               "hyperparameters" = hyperparameters,
               "accept_prob" = accept_prob,
               "EM_beta_chain" = EM_beta_chain,
               "EM_beta_estimate" = beta,
               "log_joint" = chain_log_joint,
               "beta_hat" = beta_hat,
               "gamma_hat" = gamma_hat,
               "phi_hat" = phi_hat,
               "lambda_hat" = lambda_hat,
               "alpha_hat" = alpha_hat,
               "time" = time,
               "time_vec" = cummulative_times,
               "time_vec2" = iteration_times)
    
  }else{
    
    ans = list("beta" = chain_beta,
               "gamma" = chain_gamma,
               "phi" = chain_phi,
               "lambda" = chain_lambda,
               "u" = chain_u,
               "sample_alpha" = sample_alpha,
               "hyperparameters" = hyperparameters,
               "EM_beta_chain" = EM_beta_chain,
               "EM_beta_estimate" = beta,
               "log_joint" = chain_log_joint,
               "beta_hat" = beta_hat,
               "gamma_hat" = gamma_hat,
               "phi_hat" = phi_hat,
               "lambda_hat" = lambda_hat,
               "alpha_hat" = alpha_hat,
               "time" = time,
               "time_vec" = cummulative_times,
               "time_vec2" = iteration_times)
    
  }
  
  
  return(ans)
  
}

