######################################################################
########################### DRlm-Regression ##########################
######################################################################
# =====================================================================
# Fit (Closed-form Solution)
# =====================================================================
#' @param X_list list of feature matrices from each source (each n_l x p)
#' @param y_list list of outcome vectors from each source (each n_l x 1)
#' @param loading_mat loading matrix (p x m) for m loadings of interest
#' @param X0 target feature matrix (N x p); if NULL, use all source data
#' @param intercept whether to include intercept in outcome models (default: FALSE)
#' @param intercept_loading whether to include intercept in loading models (default: FALSE)
#' @param delta regularization parameter in weight optimization (default: 0)
#' @param lambda regularization parameter in high-dimensional outcome models;
#' if "CV.min" or "CV.1se", use cross-validation to select (default: NULL)
#' @param verbose whether to print fitting progress (default: FALSE)
#' @return a list containing the following components:
#' @importFrom SIHR LF
#' @importFrom stats as.formula glm qnorm
#' @importFrom Matrix Diagonal
#' @importFrom glmnet glmnet cv.glmnet
#' @importFrom CVXR Variable Problem Minimize solve
#' @importFrom glmnet glmnet cv.glmnet
#' @example
#' set.seed(0)
#' p <- 100; L <- 2
#' A1gen <- function(rho, p) { i <- matrix(rep(1:p, each=p), p); j <- t(i); rho^abs(i - j) }
#' cov_source <- A1gen(0.6, p)
#' X1 <- MASS::mvrnorm(100, mu = rep(0,p), Sigma = cov_source)
#' X2 <- MASS::mvrnorm(100, mu = rep(0,p), Sigma = cov_source)
#' b1 <- rep(0,p); b1[1:5] <- (1:5)/20; b1[97:99] <- c(0.5, -0.5, -0.5)
#' b2 <- rep(0,p); b2[6:10] <- (1:5)/20; b2[97:99] <- 0.5*c(0.5,-0.5,-0.5)
#' Y1 <- as.numeric(X1 %*% b1 + rnorm(100))
#' Y2 <- as.numeric(X2 %*% b2 + rnorm(100))
#' cov0 <- cov_source; diag(cov0) <- 1.5; cov0[1:5,1:5] <- 0.9; diag(cov0[1:5,1:5]) <- 1.5; cov0[99:100,99:100] <- matrix(c(1.5,0.9,0.9,1.5),2)
#' X0 <- MASS::mvrnorm(100, mu = rep(0,p), Sigma = cov0)
#' loading_mat <- matrix(0, nrow = 100, ncol = 2); loading_mat[96:100,1] <- 0.4; loading_mat[99:100,2] <- 0.8 #; loading_mat <- t(loading_mat)
#' fit <- cgdro(list(X1,X2), list(Y1,Y2), X0 = X0,
#'           family = "drlm_reg", f_learner = "high_d", w_learner = "linear",
#'           loading_mat = loading_mat, intercept = FALSE,
#'           delta = 0, lambda = "CV.min", verbose = FALSE)
#' inf <- infer(fit, M = 50)
#' summary(fit, infer=inf)
#' pred <- predict(fit)
#' @export
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


  # ---- HD: Lasso init + SIHR::LF debiasing ----
  init_est <- vector("list", L)
  for (l in seq_len(L)) {
    Xc <- if (intercept) cbind(1, X_list[[l]]) else X_list[[l]]
    init_model <- .learn_f(mode = "reg", learner = "high_d", lambda_val = lambda)
    init_model$fit(Xc, y_list[[l]],intercept = intercept)
    b_init <- init_model$coef()
    sparsity <- sum(abs(b_init) > 1e-4)
    pred_lin <- as.numeric(Xc %*% b_init)
    dev <- .compute_dev(pred_lin, y_list[[l]], sparsity = sparsity)
    init_est[[l]] <- list(beta_init = b_init, dev = dev)
  }

  Sigma0 <- crossprod(X0_use) / N
  Gamma_plugin <- matrix(0, L, L)
  for (l in seq_len(L)) {
    bl <- init_est[[l]]$beta_init
    for (k in l:L) {
      bk <- init_est[[k]]$beta_init
      Gamma_plugin[l, k] <- as.numeric(t(bl) %*% Sigma0 %*% bk)
    }
  }
  Gamma_plugin[lower.tri(Gamma_plugin)] <- t(Gamma_plugin)[lower.tri(Gamma_plugin)]

  correct_mat <- matrix(0, L, L)
  Proj_array <- array(0, dim = c(L, L, d))
  for (l in seq_len(L)) {
    Xl <- X_list[[l]]
    yl <- y_list[[l]]
    for (k in seq_len(L)) {
      bk <- init_est[[k]]$beta_init
      loading_vec <- as.vector(Sigma0 %*% bk)

      Est_lk <- SIHR::LF(
        X = Xl, y = yl,
        loading.mat = loading_vec,
        model = "linear",
        intercept = intercept,
        beta.init = init_est[[l]]$beta_init,
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

  w <- opt_weight(Gamma_debias, delta = delta)

  # Debiased loadings per source on the *user-supplied* loading_mat
  debias_est <- vector("list", L)
  for (l in seq_len(L)) {
    Xl <- X_list[[l]]
    yl <- y_list[[l]]
    est_all <- SIHR::LF(
      X = Xl, y = yl,
      loading.mat = loading_mat,
      model = "linear",
      intercept = intercept,
      intercept.loading = intercept_loading,
      beta.init = init_est[[l]]$beta_init,
      verbose = verbose
    )
    debias_est[[l]] <- list(
      est.debias.vec = as.numeric(est_all$est.debias.vec),
      se.vec         = as.numeric(est_all$se.vec)
    )
  }
  est_bc <- Reduce(`+`, Map(function(wl, pt) wl * pt$est.debias.vec, w, debias_est))
  beta_plug <- Reduce(`+`, Map(function(wl, pt) wl * pt$beta_init, w, init_est))
  est_plug <- beta_plug %*% loading_mat

  return(list(
    L = L, d = d, intercept = intercept,
    X0 = X0, X0_use = X0_use,
    loading_mat = loading_mat,
    Gamma = Gamma_debias,
    Gamma_plugin = Gamma_plugin,
    weight_ = w,
    est_bc_ = as.numeric(est_bc),
    est_plug_ = as.numeric(est_plug),
    beta_plug_ = as.numeric(beta_plug),
    init_est = init_est,
    debias_est = debias_est,
    Proj_array = Proj_array,
    delta = delta,
    family = "drlm_reg"
  ))

}

# =====================================================================
# Predict on target
# =====================================================================
#' @param fit a fitted model returned by fitting
#' @return a numeric vector of predicted outcomes on target (length N)

predict_drlm_reg <- function(fit) {

  pred_plugin <- as.numeric(fit$X0_use %*% fit$beta_plug_)
  return(pred_plugin)
}

# =====================================================================
# Inference
# =====================================================================
#' @param fit a fitted model returned by fitting
#' @param M number of Monte Carlo samples (default: 500)
#' @param alpha significance level (default: 0.05)
#' @param tau variance inflation parameter (default: 0.2)
#' @param alpha_thres threshold for small eigenvalues (default: 0.01)
#' @param threshold threshold for eigenvalue truncation (default: 0)
#' @return a list containing the following components:
#' \item{CI}{an n_loading x 2 matrix of confidence intervals for each loading}

infer_drlm_reg <- function(fit, M = 500, alpha = 0.05,
                           tau = 0.2, alpha_thres = 0.01, threshold = 0) {
  # check arguments
  check_arg_drlm_reg_inf(fit, M, alpha, tau, alpha_thres, threshold, fit$delta)

  L <- fit$L
  Gamma <- fit$Gamma
  tril_idx <- which(lower.tri(Gamma, diag = TRUE), arr.ind = TRUE)
  mu_vec <- Gamma[tril_idx]

  Var_Gamma <- .compute_Var_Gamma(fit, tau = tau)
  gen_samples <- .gensamples(mu_vec, Var_Gamma, gen_size = M,
                             threshold = threshold, alpha_thres = alpha_thres)

  weight_mat <- matrix(NA_real_, M, L)


  n_loading <- nrow(fit$loading_mat)
  loading_per_source <- do.call(rbind, lapply(fit$debias_est, `[[`, "est.debias.vec"))
  ses_per_source     <- do.call(rbind, lapply(fit$debias_est, `[[`, "se.vec"))


  n_loading <- ncol(ses_per_source)
  lb_all <- matrix(Inf,  M, n_loading)
  ub_all <- matrix(-Inf, M, n_loading)
  z <- stats::qnorm(1 - alpha / 2)

  for (m in seq_len(M)) {
    Gm <- matrix(0, L, L)
    Gm[lower.tri(Gm, diag = TRUE)] <- gen_samples[m, ]
    Gm <- Gm + t(Gm) - diag(diag(Gm))

    w <- opt_weight(Gm, delta = fit$delta)
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
    CI = CI_U
  )
}



# =====================================================================
# Summary
# =====================================================================

summary_drlm_reg <- function(fit, infer = NULL, index = NULL) {
  width <- 8; per_row_coef <- 10; per_row_ci <- 5; digits_coef <- 4; digits_ci <- 4

  # ---- basic checks ----
  req <- c("weight_", "est_plug_", "est_bc_")
  missing_req <- req[!vapply(req, function(nm) !is.null(fit[[nm]]), logical(1))]
  if (length(missing_req)) {
    stop(sprintf("Missing fields in 'fit': %s", paste(missing_req, collapse = ", ")))
  }

  w         <- as.numeric(fit$weight_)     # length L
  est_plug  <- as.numeric(fit$est_plug_)   # length d
  est_bc    <- as.numeric(fit$est_bc_)     # length n_loading
  L         <- length(w)
  n_loading <- length(est_bc)


  # ---- helpers (same style as your cls summary) ----
  normalize_indices <- function(user_idx, lo_1based, hi_1based, name) {
    if (is.null(user_idx)) return(seq.int(lo_1based, hi_1based) - 1L)  # internal 0-based
    idx <- tryCatch(as.integer(user_idx), error = function(e) NA_integer_)
    if (any(is.na(idx))) stop(sprintf("%s must be integer(-coercible).", name))
    if (any(idx < lo_1based | idx > hi_1based)) {
      bad <- idx[idx < lo_1based | idx > hi_1based]
      stop(sprintf("%s out of range: %s not in [%d,%d]", name, paste(bad, collapse=","), lo_1based, hi_1based))
    }
    idx[!duplicated(idx)] - 1L
  }

  print_chunks <- function(label, indices0, values, width = 8, per_row = 10,
                           fmt = "%8.4f", header_label = "index") {
    n <- length(indices0); if (n == 0L) return()
    starts <- seq.int(1L, n, by = per_row)
    for (s in starts) {
      e <- min(s + per_row - 1L, n)
      idx_chunk  <- indices0[s:e]
      vals_chunk <- values[s:e]
      header <- paste0(sprintf("%-10s| ", header_label),
                       paste(sprintf(paste0("%", width, "d"), idx_chunk + 1L), collapse = " "))
      row    <- paste0(sprintf("%-10s| ", label),
                       paste(if (is.character(vals_chunk)) vals_chunk else sprintf(fmt, vals_chunk),
                             collapse = " "))
      cat(header, "\n", row, "\n", sep = "")
    }
  }

  dim_idx0 <- normalize_indices(index, 1L, n_loading, "index")

  # ---- print ----
  cat("Model Summary:\n")
  cat("=================================\n")

  # Weights
  cat("Fitted Weights:\n\n")
  group_idx0 <- seq_len(L) - 1L
  print_chunks("weight_", group_idx0, w,
               width = width, per_row = 10,
               fmt = sprintf("%%%d.%df", width, digits_coef),
               header_label = "group")
  cat("\n")

  cat("=================================\n")

  # Plug-in estimates (all loadings)
  cat("Fitted Plug-in Estimations:\n\n")
  print_chunks("coef_", dim_idx0, est_plug,
               width = width, per_row = per_row_coef,
               fmt = sprintf("%%%d.%df", width, digits_coef),
               header_label = "index")
  cat("\n")

  cat("=================================\n")

  # Debiased estimates (subset by index)
  cat("Fitted Debiased Estimations:\n\n")
  print_chunks("coef_", dim_idx0, est_bc[dim_idx0 + 1L],
               width = width, per_row = per_row_coef,
               fmt = sprintf("%%%d.%df", width, digits_coef),
               header_label = "index")
  cat("\n")

  # Confidence intervals (optional; from infer_drlm_reg)
  if (!is.null(infer) && !is.null(infer$CI)) {
    CI <- infer$CI
    if (!is.matrix(CI) || nrow(CI) != n_loading || ncol(CI) != 2) {
      warning("infer$CI should be an n_loading x 2 matrix (lb, ub). Skipping CI print.")
    } else {
      cat("=================================\n")
      cat("Confidence Intervals:\n\n")
      ci_sub <- CI[dim_idx0 + 1L, , drop = FALSE]
      ci_str <- paste0("(", formatC(ci_sub[, 1], format = "f", digits = digits_ci), ",",
                       formatC(ci_sub[, 2], format = "f", digits = digits_ci), ")")
      print_chunks("CI", dim_idx0, ci_str,
                   width = max(width, 14), per_row = per_row_ci,
                   fmt = "%s", header_label = "index")
      cat("\n")
    }
  } else {
    cat("Confidence Intervals not provided. Run infer_drlm_reg() and pass its result via infer=.\n")
  }

  invisible(NULL)
}




