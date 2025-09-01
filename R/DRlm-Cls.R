######################################################################
######################### DRlm-Classification ########################
######################################################################
source("R/Utility.R")
# =====================================================================
# Fit (Optimistic Gradient Mirror Prox)
# =====================================================================
fit_drlm_cls <- function(X_list, y_list, X0 = NULL,
                         prob_learner = "xgb", density_learner = "linear",
                         split = TRUE, intercept = FALSE,
                         theta_init = NULL, gamma_init = NULL, eta_init = NULL,
                         max_iter = 1000, tol = 1e-6, check_dual = FALSE,
                         conv_dual = FALSE, verbose = FALSE,
                         seed = 123) {
  set.seed(seed)
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
  proba_params_list   <- replicate(L, NULL, simplify = FALSE)
  density_params_list <- replicate(L, NULL, simplify = FALSE)

  # check arguments
  check_arg_drlm_cls_fit(X_list, y_list, X0, prob_learner, density_learner,
                         split, intercept, theta_init, gamma_init, eta_init,
                         max_iter, tol, check_dual, conv_dual, verbose, seed)

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
                           parallel = FALSE, n_workers = 4, diag = TRUE) {
  # check arguments
  check_arg_drlm_cls_inf(fit, index, M, alpha, parallel, n_workers, diag)

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


