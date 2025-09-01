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

  # check arguments
  check_arg_drlm_reg(X_list, y_list, loading_mat, X0, intercept,
                     intercept_loading, delta, lambda, verbose)

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
  # check arguments
  check_arg_drlm_reg_inf(fit, M, alpha, tau, alpha_thres, threshold, delta)

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



