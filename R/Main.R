######################### Main Functions Used in CGDRO #################################
###########################################################################################
source("Utility.R")
######################################################################
######################### DRlm-Classification ########################
######################################################################
# =====================================================================
# Fit (Optimistic Gradient Mirror Prox)
# =====================================================================
fit_drlm_cls <- function(X_list, y_list, X0 = NULL,
                         prob_learner = "xgb", density_learner = "linear",
                         split = TRUE, intercept = FALSE,
                         proba_params_list = NULL, density_params_list = NULL,
                         theta_init = NULL, gamma_init = NULL, eta_init = NULL,
                         max_iter = 1000, tol = 1e-6, check_dual = FALSE,
                         conv_dual = FALSE, verbose = FALSE, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  # center per-domain; build X0
  X_list <- lapply(X_list, function(Xi) { Xi <- as.matrix(Xi); sweep(Xi, 2, colMeans(Xi), "-") })
  if (is.null(X0)) X0 <- do.call(rbind, X_list)
  X0 <- as.matrix(X0); X0 <- sweep(X0, 2, colMeans(X0), "-")
  
  if (intercept) {
    X_list <- lapply(X_list, function(Xi) cbind(1, Xi))
    X0 <- cbind(1, X0)
  }
  L <- length(X_list)
  d <- ncol(X_list[[1]])
  num_class <- length(levels(as.factor(y_list[[1]])))  # ensure factor levels
  if (is.null(proba_params_list))   proba_params_list   <- replicate(L, NULL, simplify = FALSE)
  if (is.null(density_params_list)) density_params_list <- replicate(L, NULL, simplify = FALSE)
  
  # learners
  pd <- .fit_proba_density(X_list, y_list, X0, prob_learner, density_learner,
                           split, proba_params_list, density_params_list, seed)
  mu_list <- .compute_mu_list(X_list, y_list, X0, pd$probaX_list, pd$probaX0_list, pd$omegaX_list, num_class, d)
  
  # init
  K <- num_class - 1
  if (is.null(theta_init)) theta <- rep(0, d * K) else theta <- theta_init
  if (is.null(gamma_init)) gamma <- rep(1/L, L) else gamma <- gamma_init
  theta_bar <- theta; gamma_bar <- gamma
  eta <- if (is.null(eta_init)) sqrt(2) else eta_init
  a <- 1.2; b <- log(L); Z_cumsum <- 0
  primal <- .compute_primal(theta, mu_list, X0, d)
  log_message <- character(0)
  
  for (iter in seq_len(max_iter)) {
    g1 <- .compute_grad(theta_bar, gamma_bar, mu_list, X0, d)
    theta_bar <- theta - (eta/a)*g1$grad_theta
    gamma_bar <- gamma_bar * exp((eta/b) * g1$grad_gamma); gamma_bar <- gamma_bar/sum(gamma_bar)
    
    g2 <- .compute_grad(theta_bar, gamma_bar, mu_list, X0, d)
    theta_curr <- theta_bar - (eta/a)*g2$grad_theta
    gamma_curr <- gamma_bar * exp((eta/b) * g2$grad_gamma); gamma_curr <- gamma_curr/sum(gamma_curr)
    
    Z <- a * (sum((theta_bar - theta_curr)^2) + sum((theta_bar - theta)^2)) +
      b * (sum(abs(gamma_bar - gamma_curr))^2 + sum(abs(gamma_bar - gamma))^2)
    Z_cumsum <- Z_cumsum + Z/(5*eta^2)
    eta <- sqrt(2)/sqrt(1 + Z_cumsum)
    
    theta <- theta_curr; gamma <- gamma_curr
    primal_curr <- .compute_primal(theta, mu_list, X0, d)
    
    if (iter %% 20 == 0) {
      if (check_dual) {
        dual <- .compute_dual_value(theta, gamma, mu_list, X0, d)
        msg <- sprintf("Iter %d | Diff primal: %.6f | Dual gap: %.6f", iter, abs(primal - primal_curr), abs(primal - dual))
      } else {
        msg <- sprintf("Iter %d | Diff primal: %.6f", iter, abs(primal - primal_curr))
      }
      log_message <- c(log_message, msg)
      if (isTRUE(verbose)) cat(msg, "\n")
    }
    
    if (abs(primal_curr - primal) < tol) {
      if (conv_dual) {
        dual <- .compute_dual_value(theta, gamma, mu_list, X0, d)
        if (abs(primal_curr - dual) < 1e-3) { if (isTRUE(verbose)) cat("Converged (dual).\n"); break }
      } else { if (isTRUE(verbose)) cat("Converged.\n"); break }
    }
    primal <- primal_curr
  }
  theta_array <- matrix(theta, nrow = d, byrow = FALSE)
  
  
  list(
    theta = theta_array, gamma = gamma, d = d, K = K, num_class = num_class,
    X0 = X0, X_list = X_list, y_list = y_list, L = L, intercept = intercept,
    probaX_list = pd$probaX_list, probaX0_list = pd$probaX0_list, omegaX_list = pd$omegaX_list,
    mu_list = mu_list, log_message = log_message
  )
}

# =====================================================================
# Predict on target
# =====================================================================
predict_proba_drlm_cls <- function(fit) {
  theta_mat <- matrix(fit$theta, nrow = fit$d, byrow = FALSE)
  logits <- fit$X0 %*% theta_mat
  logits_max <- apply(logits, 1, max)
  stable <- sweep(logits, 1, logits_max, "-")
  exp_terms <- exp(stable)
  proba_red <- sweep(exp_terms, 1, 1 + rowSums(exp_terms), "/")  # (n, K)
  cbind(1 - rowSums(proba_red), proba_red)                       # (n, C)
}

# =====================================================================
# Inference 
# =====================================================================

infer_drlm_cls <- function(fit, index = 1, M = 200, alpha = 0.05,
                           parallel = FALSE, n_workers = 4, diag = TRUE, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  caches <- .prepare_inference(fit, diag = diag)
  z <- qnorm(1 - alpha/2)
  
  one_draw <- function() {
    mu_resample_list <- Map(function(mu, cov) mvrnorm(1, mu = mu, Sigma = cov),
                            fit$mu_list, caches$mu_cov_list)
    rs <- .solve_resample(fit$theta, caches, mu_resample_list, fit$X0, fit$d, diag)
    V <- .compute_variance_resample(rs$gamma, caches)
    lb <- rs$theta - z * sqrt(diag(V))
    ub <- rs$theta + z * sqrt(diag(V))
    list(theta = rs$theta, gamma = rs$gamma, lb = lb, ub = ub)
  }
  
  res <- if (parallel) {
    parallel::mclapply(seq_len(M), function(i) one_draw(), mc.cores = n_workers)
  } else {
    lapply(seq_len(M), function(i) one_draw())
  }
  
  CI_lb_U <- Reduce(pmin, lapply(res, `[[`, "lb"))
  CI_ub_U <- Reduce(pmax, lapply(res, `[[`, "ub"))
  d <- fit$d; K <- fit$K
  CI_Union <- array(NA_real_, dim = c(d, 2, K))
  for (k in seq_len(K)) {
    rows <- ((k-1)*d + 1):(k*d)
    CI_Union[, 1, k] <- CI_lb_U[rows]
    CI_Union[, 2, k] <- CI_ub_U[rows]
  }
  # turn CI_Union into a list for each label K
  CI_Union_list <- lapply(seq_len(K), function(k) CI_Union[, , k])
  
  list(
    theta_M = lapply(res, `[[`, "theta"),
    gamma_M = lapply(res, `[[`, "gamma"),
    CI_lb_M = lapply(res, `[[`, "lb"),
    CI_ub_M = lapply(res, `[[`, "ub"),
    CI_lb_U = CI_lb_U,
    CI_ub_U = CI_ub_U,
    CI_U = CI_Union_list,
    CI_index = CI_Union[index, , , drop = FALSE]
  )
}


######################################################################
########################### DRlm-Regression ##########################
######################################################################
# =====================================================================
# Fit (Closed-form Solution)
# =====================================================================

fit_drlm_reg <- function(X_list, y_list, loading_mat,
                         X0 = NULL,
                         intercept = FALSE,
                         intercept_loading = FALSE,
                         delta = 0,
                         lambda = NULL,
                         verbose = FALSE) {
  stopifnot(length(X_list) == length(y_list))
  L <- length(X_list)
  
  X_list <- lapply(X_list, function(Xi) { Xi <- as.matrix(Xi); sweep(Xi, 2, colMeans(Xi), "-") })
  y_list <- lapply(y_list, as.numeric)
  
  d <- ncol(X_list[[1]]) + if (intercept) 1 else 0
  n_loading <- nrow(loading_mat)
  
  if (is.null(X0)) X0 <- do.call(rbind, X_list)
  X0 <- sweep(as.matrix(X0), 2, colMeans(as.matrix(X0)), "-")
  X0_use <- if (intercept) cbind(1, X0) else X0
  N <- nrow(X0_use)
  
  ns <- vapply(X_list, nrow, 1L)
  max_n <- max(ns)
  high_dim <- (max_n < 6 * d)
  if (verbose) cat(if (high_dim) "start high-dimensional fitting-----\n" else "start low-dimensional fitting-----\n")
  
  if (high_dim) {
    # ---- HD: Lasso init + SIHR::LF debiasing ----
    fits_info <- vector("list", L)
    for (l in seq_len(L)) {
      Xc <- if (intercept) cbind(1, X_list[[l]]) else X_list[[l]]
      b_init <- .train_lasso(X_list[[l]], y_list[[l]], intercept = intercept, lambda_val = lambda)
      sparsity <- sum(abs(b_init) > 1e-4)
      pred_lin <- as.numeric(Xc %*% b_init)
      dev <- .compute_dev(pred_lin, y_list[[l]], sparsity = sparsity)
      fits_info[[l]] <- list(beta_init = b_init, dev = dev)
    }
    
    Sigma0 <- crossprod(X0_use) / N
    Gamma_plugin <- matrix(0, L, L)
    for (l in seq_len(L)) {
      bl <- fits_info[[l]]$beta_init
      for (k in l:L) {
        bk <- fits_info[[k]]$beta_init
        Gamma_plugin[l, k] <- as.numeric(t(bl) %*% Sigma0 %*% bk)
      }
    }
    Gamma_plugin[lower.tri(Gamma_plugin)] <- t(Gamma_plugin)[lower.tri(Gamma_plugin)]
    
    correct_mat <- matrix(0, L, L)
    Proj_array <- array(0, dim = c(L, L, d))
    for (l in seq_len(L)) {
      Xl <- if (intercept) cbind(1, X_list[[l]]) else X_list[[l]]
      yl <- y_list[[l]]
      for (k in seq_len(L)) {
        bk <- fits_info[[k]]$beta_init
        loading_vec <- as.vector(Sigma0 %*% bk)
        
        Est_lk <- SIHR::LF(
          X = Xl, y = yl,
          loading.mat = loading_vec,
          model = "linear",
          intercept = intercept,
          beta.init = fits_info[[l]]$beta_init,
          verbose = verbose
        )
        correct_mat[l, k] <- as.numeric(Est_lk$est.debias.vec - Est_lk$est.plugin.vec)
        Proj_array[l, k, ] <- as.numeric(Est_lk$proj.mat)
      }
    }
    
    Gamma_debias <- Gamma_plugin
    for (l in seq_len(L)) for (k in l:L) {
      Gamma_debias[l, k] <- Gamma_plugin[l, k] + correct_mat[l, k] + correct_mat[k, l]
    }
    Gamma_debias[lower.tri(Gamma_debias)] <- t(Gamma_debias)[lower.tri(Gamma_debias)]
    
    w <- .opt_weight_cvxr(Gamma_debias, delta = delta)
    
    # Debiased loadings per source on the *user-supplied* loading_mat
    Points_info <- vector("list", L)
    for (l in seq_len(L)) {
      Xl <- if (intercept) cbind(1, X_list[[l]]) else X_list[[l]]
      yl <- y_list[[l]]
      est_all <- SIHR::LF(
        X = Xl, y = yl,
        loading.mat = loading_mat,
        model = "linear",
        intercept = intercept,
        intercept.loading = intercept_loading,
        beta.init = fits_info[[l]]$beta_init,
        verbose = verbose
      )
      Points_info[[l]] <- list(
        est.debias.vec = as.numeric(est_all$est.debias.vec),
        se.vec         = as.numeric(est_all$se.vec)
      )
    }
    loading_coef <- Reduce(`+`, Map(function(wl, pt) wl * pt$est.debias.vec, w, Points_info))
    
    return(list(
      mode = "high_dim",
      L = L, d = d, intercept = intercept,
      X0 = X0, X0_use = X0_use,
      loading_mat = loading_mat,
      max_n_X_list = max_n,
      Gamma = Gamma_debias,
      Gamma_plugin = Gamma_plugin,
      weight_ = w,
      loading_coef_ = as.numeric(loading_coef),
      fits_info = fits_info,
      Points_info = Points_info,
      Proj_array = Proj_array
    ))
  } else {
    # ---- LD: OLS + plugin ----
    beta_list <- matrix(NA_real_, nrow = L, ncol = d)
    for (l in seq_len(L)) {
      Xc <- if (intercept) cbind(1, X_list[[l]]) else X_list[[l]]
      y  <- y_list[[l]]
      XtX <- crossprod(Xc); Xty <- crossprod(Xc, y)
      beta_list[l, ] <- as.numeric(solve(XtX, Xty))
    }
    
    Sigma0 <- crossprod(X0_use) / N
    Gamma <- matrix(0, L, L)
    for (l in seq_len(L)) for (k in seq_len(L)) {
      bl <- beta_list[l, ]; bk <- beta_list[k, ]
      Gamma[l, k] <- as.numeric(t(bl) %*% Sigma0 %*% bk)
    }
    
    dev_vec <- numeric(L)
    var_loading_list <- vector("list", L)
    var_loading_mat  <- matrix(NA_real_, nrow = L, ncol = n_loading)
    for (l in seq_len(L)) {
      Xc <- if (intercept) cbind(1, X_list[[l]]) else X_list[[l]]
      y  <- y_list[[l]]
      r  <- y - as.numeric(Xc %*% beta_list[l, ])
      sigma2 <- sum(r^2) / (nrow(Xc) - d)
      Vb <- sigma2 * solve(crossprod(Xc))
      Vload <- loading_mat %*% Vb %*% t(loading_mat)
      var_loading_list[[l]] <- Vload
      var_loading_mat[l, ]  <- diag(Vload)
      dev_vec[l] <- sigma2
    }
    
    w <- .opt_weight_cvxr(Gamma, delta = delta)
    
    coef_ <- as.numeric(t(beta_list) %*% w)
    loading_beta_list <- t(apply(beta_list, 1, function(b) loading_mat %*% b))
    loading_coef_ <- as.numeric(t(loading_beta_list) %*% w)
    
    return(list(
      mode = "low_dim",
      L = L, d = d, intercept = intercept,
      X0 = X0, X0_use = X0_use,
      loading_mat = loading_mat,
      max_n_X_list = max_n,
      beta_list = beta_list,
      Gamma = Gamma,
      weight_ = w,
      coef_ = coef_,
      loading_beta_list = loading_beta_list,
      loading_coef_ = loading_coef_,
      dev_vec = dev_vec,
      var_loading_list = var_loading_list,
      var_loading_mat = var_loading_mat
    ))
  }
}

# =====================================================================
# Predict on target
# =====================================================================

predict_drlm_reg <- function(fit) {
  if (fit$mode == "low_dim") {
    as.numeric(fit$X0_use %*% fit$coef_)
  } else {
    stop("High-dimensional mode only provides loading coefficients (no full β prediction).")
  }
}

# =====================================================================
# Inference 
# =====================================================================

infer_drlm_reg <- function(fit, M = 500, alpha = 0.05,
                           tau = 0.2, alpha_thres = 0.01, threshold = 0,
                           delta = 0) {
  L <- fit$L
  Gamma <- fit$Gamma
  tril_idx <- which(lower.tri(Gamma, diag = TRUE), arr.ind = TRUE)
  mu_vec <- Gamma[tril_idx]
  
  Var_Gamma <- .compute_Var_Gamma(fit, tau = tau)
  gen_samples <- .gensamples(mu_vec, Var_Gamma, gen_size = M,
                             threshold = threshold, alpha_thres = alpha_thres)
  
  weight_mat <- matrix(NA_real_, M, L)
  
  if (fit$mode == "low_dim") {
    loading_per_source <- fit$loading_beta_list      # (L x n_loading)
    ses_per_source     <- sqrt(pmax(fit$var_loading_mat, 0))
  } else {
    n_loading <- nrow(fit$loading_mat)
    loading_per_source <- do.call(rbind, lapply(fit$Points_info, `[[`, "est.debias.vec"))
    ses_per_source     <- do.call(rbind, lapply(fit$Points_info, `[[`, "se.vec"))
  }
  
  n_loading <- ncol(ses_per_source)
  lb_all <- matrix(Inf,  M, n_loading)
  ub_all <- matrix(-Inf, M, n_loading)
  z <- stats::qnorm(1 - alpha / 2)
  
  for (m in seq_len(M)) {
    Gm <- matrix(0, L, L)
    Gm[lower.tri(Gm, diag = TRUE)] <- gen_samples[m, ]
    Gm <- Gm + t(Gm) - diag(diag(Gm))
    
    w <- .opt_weight_cvxr(Gm, delta = delta)
    weight_mat[m, ] <- w
    
    gen_coef <- as.numeric(t(loading_per_source) %*% w)
    gen_se   <- as.numeric(t(ses_per_source) %*% w)
    
    lb_all[m, ] <- gen_coef - z * gen_se
    ub_all[m, ] <- gen_coef + z * gen_se
  }
  
  CI_lb_U <- apply(lb_all, 2, min, na.rm = TRUE)
  CI_ub_U <- apply(ub_all, 2, max, na.rm = TRUE)
  CI_U <- cbind(CI_lb_U, CI_ub_U)
  
  list(
    CI = CI_U,
    weights_resampled = weight_mat,
    Var_Gamma = Var_Gamma
  )
}


######################################################################
################################ DRoL ################################
######################################################################
# =====================================================================
# Fit (Closed-form Solution)
# =====================================================================

fit_drol <- function(X_list, y_list, X0 = NULL,
                     outcome_learner = "xgb",
                     density_learner = "xgb",
                     intercept = FALSE,
                     outcome_params_list = NULL,
                     density_params_list = NULL,
                     seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  # center each source; keep y numeric
  X_list <- lapply(X_list, function(Xi) { Xi <- as.matrix(Xi); sweep(Xi, 2, colMeans(Xi), "-") })
  y_list <- lapply(y_list, function(yi) as.numeric(yi))
  
  L <- length(X_list)
  stopifnot(L == length(y_list))
  nls <- vapply(X_list, nrow, 1L)
  
  # target features
  if (is.null(X0)) X0 <- do.call(rbind, X_list)
  X0 <- as.matrix(X0)
  X0 <- sweep(X0, 2, colMeans(X0), "-")
  N <- nrow(X0)
  
  # optional intercept column
  if (intercept) {
    X_list <- lapply(X_list, function(Xi) cbind(1, Xi))
    X0 <- cbind(1, X0)
  }
  
  d <- ncol(X0)
  
  if (is.null(outcome_params_list)) outcome_params_list <- replicate(L, NULL, simplify = FALSE)
  if (is.null(density_params_list)) density_params_list <- replicate(L, NULL, simplify = FALSE)
  
  # -------- Plug-in Γ (using full-source outcome models) --------
  source_full_models <- vector("list", L)
  pred_full_mat <- matrix(0, N, L)
  
  for (l in seq_len(L)) {
    om <- .fit_outcome(X_list[[l]], y_list[[l]], learner = outcome_learner,
                       params = outcome_params_list[[l]], seed = seed)
    source_full_models[[l]] <- om
    pred_full_mat[, l] <- om$predict(X0)
  }
  Gamma_plug <- crossprod(pred_full_mat) / N  # t(P) %*% P / N
  
  # -------- Sample-split models (A/B) for bias-correction --------
  source_A_models <- vector("list", L)
  source_B_models <- vector("list", L)
  density_A_models <- vector("list", L)
  density_B_models <- vector("list", L)
  
  for (l in seq_len(L)) {
    half_l <- floor(nls[l] / 2)
    idxA <- seq_len(half_l)
    idxB <- (half_l + 1):nls[l]
    
    source_A_models[[l]] <- .fit_outcome(X_list[[l]][idxA, , drop = FALSE], y_list[[l]][idxA],
                                         learner = outcome_learner,
                                         params = outcome_params_list[[l]], seed = seed)
    source_B_models[[l]] <- .fit_outcome(X_list[[l]][idxB, , drop = FALSE], y_list[[l]][idxB],
                                         learner = outcome_learner,
                                         params = outcome_params_list[[l]], seed = seed)
    density_A_models[[l]] <- .fit_density(X_list[[l]][idxA, , drop = FALSE], X0,
                                          learner = density_learner,
                                          params = density_params_list[[l]], seed = seed)
    density_B_models[[l]] <- .fit_density(X_list[[l]][idxB, , drop = FALSE], X0,
                                          learner = density_learner,
                                          params = density_params_list[[l]], seed = seed)
  }
  
  # -------- Bias-corrected Γ --------
  Gamma_corr <- Gamma_plug
  for (k in seq_len(L)) {
    fkA <- source_A_models[[k]]$predict
    fkB <- source_B_models[[k]]$predict
    wkA <- density_A_models[[k]]$predict
    wkB <- density_B_models[[k]]$predict
    half_k <- floor(nls[k] / 2)
    kA <- seq_len(half_k); kB <- (half_k + 1):nls[k]
    
    for (l in seq_len(L)) {
      flA <- source_A_models[[l]]$predict
      flB <- source_B_models[[l]]$predict
      wlA <- density_A_models[[l]]$predict
      wlB <- density_B_models[[l]]$predict
      half_l <- floor(nls[l] / 2)
      lA <- seq_len(half_l); lB <- (half_l + 1):nls[l]
      
      num1A <- .bias_correct_term(fkA, flA, wlA, X_list[[l]][lB, , drop = FALSE], y_list[[l]][lB])
      num2A <- .bias_correct_term(flA, fkA, wkA, X_list[[k]][kB, , drop = FALSE], y_list[[k]][kB])
      num1B <- .bias_correct_term(fkB, flB, wlB, X_list[[l]][lA, , drop = FALSE], y_list[[l]][lA])
      num2B <- .bias_correct_term(flB, fkB, wkB, X_list[[k]][kA, , drop = FALSE], y_list[[k]][kA])
      
      Gamma_corr[k, l] <- Gamma_corr[k, l] - (num1A + num2A + num1B + num2B) / 2
    }
  }
  Gamma_corr <- (t(Gamma_corr) + Gamma_corr) / 2  # symmetrize
  
  list(
    # meta
    outcome_learner = outcome_learner,
    density_learner = density_learner,
    intercept = intercept,
    L = L,
    d = d,
    # data cache
    X0 = X0,
    # predictions + estimators
    pred_full_mat = pred_full_mat,
    Gamma_plug = Gamma_plug,
    Gamma_corr = Gamma_corr
  )
}

# =====================================================================
# Predict on target
# =====================================================================

predict_drol <- function(fit, bias_correct = TRUE, priors = NULL,
                         ridge = 1e-8, solver = c("ECOS", "SCS")) {
  if (!requireNamespace("CVXR", quietly = TRUE)) {
    stop("Package 'CVXR' is required. Please install.packages('CVXR').")
  }
  solver <- match.arg(solver)
  Gamma <- if (bias_correct) fit$Gamma_corr else fit$Gamma_plug
  L <- fit$L
  P <- fit$pred_full_mat
  
  q <- CVXR::Variable(L)
  G <- Gamma + ridge * diag(L)
  
  # CHANGED: use sum_entries() instead of sum()
  constraints <- list(CVXR::sum_entries(q) == 1, q >= 0)
  if (!is.null(priors)) {
    q0  <- as.numeric(priors[[1]])
    rho <- as.numeric(priors[[2]])
    if (length(q0) != L) stop("prior_weight length must equal number of sources L.")
    if (rho < 0) stop("rho must be nonnegative.")
    constraints <- c(constraints, list(CVXR::p_norm(q - q0, 2) <= rho))
  }
  
  obj <- CVXR::Minimize(CVXR::quad_form(q, G))
  prob <- CVXR::Problem(obj, constraints)
  
  res <- try(CVXR::solve(prob, solver = solver), silent = TRUE)
  if (inherits(res, "try-error") || isTRUE(res$status %in% c("infeasible", "unbounded"))) {
    res <- CVXR::solve(prob, solver = "SCS")
  }
  
  q_opt <- as.numeric(res$getValue(q))
  q_opt[is.na(q_opt)] <- 0
  q_opt[q_opt < 0] <- 0
  s <- sum(q_opt)
  if (s <= 0) stop("Optimization failed: weights degenerate.")
  q_opt <- q_opt / s
  
  pred <- as.numeric(P %*% q_opt)
  list(weight_ = q_opt, pred = pred, status = res$status, value = res$value)
}