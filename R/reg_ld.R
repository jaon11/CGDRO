######################################################################
########################### Regression (low-dim) #####################
######################################################################
# =====================================================================
# Fit (Closed-form Solution)
# =====================================================================
fit_reg_ld <- function(X_list, y_list,
                         X0 = NULL,
                         loss_type = c("reward", "squaredloss", "regret"),
                         intercept = FALSE,
                         delta = 0,
                         verbose = FALSE) {
  stopifnot(length(X_list) == length(y_list))
  L <- length(X_list)

  X_list <- lapply(X_list, function(Xi) { Xi <- as.matrix(Xi); sweep(Xi, 2, colMeans(Xi), "-") })
  y_list <- lapply(y_list, as.numeric)

  d <- ncol(X_list[[1]]) + if (intercept) 1 else 0

  X_list_use <- lapply(X_list, function(Xi) if (intercept) cbind(1, Xi) else Xi)

  if (is.null(X0)) X0 <- do.call(rbind, X_list)
  X0 <- sweep(as.matrix(X0), 2, colMeans(as.matrix(X0)), "-")
  X0_use <- if (intercept) cbind(1, X0) else X0
  N <- nrow(X0_use)




  # check arguments
  check_arg_reg_ld(X_list, y_list, X0, loss_type,
                   intercept, delta, verbose)

  if (verbose) cat( "start  fitting-----\n")




  dev_vec <- rep(0, L)
  init_est <- vector("list", L)
  for (l in seq_len(L)) {
    Xc <- if (intercept) cbind(1, X_list[[l]]) else X_list[[l]]
    init_model <- lm(y_list[[l]] ~ Xc - 1)
    b_init <- coef(init_model)


    y_l <- y_list[[l]]
    ndev <- if (intercept) (nrow(Xc) - d - 1) else (nrow(Xc) - d)
    dev <- sum((y_l - Xc %*% b_init)^2) / ndev
    dev_vec[l] <- dev
    var_l <- dev * solve(t(Xc) %*% Xc)
    init_est[[l]] <- list(beta_init = b_init, dev=diag(var_l))
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


  w <- .opt_weight(Gamma_plugin, delta = delta, loss_type = loss_type, dev_vec = dev_vec)
  est <- Reduce(`+`, Map(function(wl, pt) wl * pt$beta_init, w, init_est))


  return(list(
    L = L, d = d, intercept = intercept,
    X0 = X0, X0_use = X0_use,
    X_list = X_list, y_list = y_list,
    Gamma = Gamma_plugin,
    weight_ = w,
    est_ = est,
    init_est = init_est,
    delta = delta,
    dev_vec = dev_vec,
    loss_type = loss_type,
    family = "reg_ld"
  ))

}

# =====================================================================
# Predict on target
# =====================================================================
predict_reg_ld <- function(fit, X=NULL) {
  if (is.null(X)) {
    X <- fit$X0_use
  } else {
    if (dim(X)[2] != fit$d) {
      stop("Dimension of X does not match the fitted model.")
    }
  }
  pred <- as.numeric(X %*% fit$est_)
  return(pred)
}

# =====================================================================
# Inference
# =====================================================================
infer_reg_ld <- function(fit, M = 500, alpha = 0.05,
                           tau = 0.2, alpha_thres = 0.01, threshold = 0) {
  # check arguments
  check_arg_reg_inf(fit, M, alpha, tau, alpha_thres, threshold, fit$delta)

  if (fit$loss_type != "reward") {
    stop("Currently inference is only available for loss_type = 'reward'")
  }

  L <- fit$L
  Gamma <- fit$Gamma
  dev_vec <- fit$dev_vec
  tril_idx <- which(lower.tri(Gamma, diag = TRUE), arr.ind = TRUE)
  mu_vec <- Gamma[tril_idx]

  Var_Gamma <- .compute_Var_Gamma_ld(fit, tau = tau)
  gen_samples <- .gensamples(mu_vec, Var_Gamma, gen_size = M,
                             threshold = threshold, alpha_thres = alpha_thres)

  weight_mat <- matrix(NA_real_, M, L)

  est_per_source <- sapply(fit$init_est, function(x) x$beta_init)  # Shape: (d, L)
  ses_per_source <- sapply(fit$init_est, function(x) sqrt(x$dev))  # Shape: (d, L)
  lb_all <- matrix(Inf,  M, fit$d)
  ub_all <- matrix(-Inf, M, fit$d)
  z <- stats::qnorm(1 - alpha / 2)

  for (m in seq_len(M)) {
    Gm <- matrix(0, L, L)
    Gm[lower.tri(Gm, diag = TRUE)] <- gen_samples[m, ]
    Gm <- Gm + t(Gm) - diag(diag(Gm))

    w <- .opt_weight(Gm, delta = fit$delta, dev_vec = fit$dev_vec)
    weight_mat[m, ] <- w

    gen_coef <- as.numeric((est_per_source) %*% w)
    gen_se   <- as.numeric((ses_per_source) %*% w)

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
summary_reg_ld <- function(fit, infer = NULL, index = NULL) {
  width <- 8; per_row_coef <- 10; per_row_ci <- 5; digits_coef <- 4; digits_ci <- 4

  # ---- basic checks ----
  req <- c("weight_", "est_")
  missing_req <- req[!vapply(req, function(nm) !is.null(fit[[nm]]), logical(1))]
  if (length(missing_req)) {
    stop(sprintf("Missing fields in 'fit': %s", paste(missing_req, collapse = ", ")))
  }

  w         <- as.numeric(fit$weight_)     # length L
  est    <- as.numeric(fit$est_)     # length d
  L         <- length(w)



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

  dim_idx0 <- normalize_indices(index, 1L, fit$d, "index")

  # ---- print ----
  cat("Model Summary:\n")
  cat("=================================\n")

  # Weights
  cat("CGDRO Aggregated Weights:\n\n")
  group_idx0 <- seq_len(L) - 1L
  print_chunks("weight_", group_idx0, w,
               width = width, per_row = 10,
               fmt = sprintf("%%%d.%df", width, digits_coef),
               header_label = "group")
  cat("\n")

  cat("=================================\n")

  # estimates
  cat("CGDRO Aggregated Estimators:\n\n")
  print_chunks("coef_", dim_idx0, est,
               width = width, per_row = per_row_coef,
               fmt = sprintf("%%%d.%df", width, digits_coef),
               header_label = "index")
  cat("\n")


  # Confidence intervals (optional; from infer_reg_ld)
  if (!is.null(infer) && !is.null(infer$CI)) {
    CI <- infer$CI
    if (!is.matrix(CI) || nrow(CI) != fit$d || ncol(CI) != 2) {
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
    cat("Confidence Intervals not provided. Run infer_reg_ld() and pass its result via infer=.\n")
  }

  invisible(NULL)
}




