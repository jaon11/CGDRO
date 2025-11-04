######################################################################
######################### DRlm-Classification ########################
######################################################################
# =====================================================================
# Fit (Optimistic Gradient Mirror Prox)
# =====================================================================
fit_cls <- function(X_list, y_list, X0 = NULL,
                         f_learner = "xgb", w_learner = "linear",
                         split = TRUE, max_iter = 1000, tol = 1e-6, check_dual = FALSE,
                         verbose = FALSE, seed = 123) {
  set.seed(seed)
  # center per-domain; build X0
  X_list <- lapply(X_list, function(Xi) { Xi <- as.matrix(Xi)})
  if (is.null(X0)) X0 <- do.call(rbind, X_list)
  X0 <- as.matrix(X0)

  L <- length(X_list)
  d <- ncol(X_list[[1]])
  num_class <- length(levels(as.factor(y_list[[1]])))  # ensure factor levels

  proba_params_list   <- replicate(L, NULL, simplify = FALSE)
  density_params_list <- replicate(L, NULL, simplify = FALSE)

  # check arguments
  check_arg_cls_fit(X_list, y_list, X0, f_learner, w_learner,
                         split, max_iter, tol, check_dual, verbose, seed)

  # learners
  if (verbose) cat("Fitting source models...\n")
  probaX_list <- list(); probaX0_list <- list(); omegaX_list <- list()
  for (i in seq_len(L)) {
    # learn f:
    if (split){
      split_info <- .split_data(X_list[[i]], y_list[[i]], seed = seed)
      XA <- as.matrix(split_info$A$X); yA <- as.vector(split_info$A$y); XB <- as.matrix(split_info$B$X); yB <- as.vector(split_info$B$y)
      modelA <- .learn_f(mode = 'cls', learner = f_learner)
      modelA$fit(XA, yA)
      modelB <- .learn_f(mode = 'cls', learner = f_learner)
      modelB$fit(XB, yB)
      fB <- modelA$predict_proba(XB)
      fA <- modelB$predict_proba(XA)
      probaX <- matrix(0, nrow = nrow(X_list[[i]]), ncol = num_class)
      probaX[split_info$idx_A, ] <- fA
      probaX[split_info$idx_B, ] <- fB
      probaX0 <- (modelA$predict_proba(X0) + modelB$predict_proba(X0)) / 2
    } else {
      model <- .learn_f(mode = 'cls', learner = f_learner)
      model$fit(X_list[[i]], y_list[[i]])
      probaX <- model$predict_proba(X_list[[i]])
      probaX0 <- model$predict_proba(X0)
    }

    # learn w:
    model_w <- .learn_w(learner = w_learner)
    model_w$fit(X_list[[i]],X0)
    omegaX <- model_w$predict(X_list[[i]])



    probaX_list[[i]] <- probaX
    probaX0_list[[i]] <- probaX0
    omegaX_list[[i]] <- omegaX
  }



  mu_list <- .compute_mu_list(X_list, y_list, X0, probaX_list, probaX0_list, omegaX_list, num_class, d)

  # init
  if (verbose) cat("Optimizing CGDRO objective via OGMP...\n")
  K <- num_class - 1
  theta <- rep(0, d * K)
  gamma <- rep(1/L, L)
  theta_bar <- theta; gamma_bar <- gamma
  eta <- sqrt(2)
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

    if (iter %% 50 == 0) {
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
      if (isTRUE(verbose)) cat("Converged.\n"); break
    }
    primal <- primal_curr
  }
  if (verbose) cat("Finished optimization.\n")
  theta_array <- matrix(theta, nrow = d, byrow = FALSE)


  list(
    coef_ = theta_array, weight_ = gamma, d = d, K = K, num_class = num_class,
    X0 = X0, X_list = X_list, y_list = y_list, L = L,
    probaX_list = probaX_list, probaX0_list = probaX0_list, omegaX_list = omegaX_list,
    mu_list = mu_list, log_message = log_message, family = "cls"
  )
}

# =====================================================================
# Predict on target
# =====================================================================
predict_cls <- function(fit, X=NULL) {
  if (is.null(X)) {
    X <- fit$X0
  } else {
    if (dim(X)[2] != fit$d) {
      stop("Dimension of X does not match the fitted model.")
    }
  }

  theta_mat <- matrix(fit$coef_, nrow = fit$d, byrow = FALSE)
  logits <- X %*% theta_mat
  logits_max <- apply(logits, 1, max)
  stable <- sweep(logits, 1, logits_max, "-")
  exp_terms <- exp(stable)
  proba_red <- sweep(exp_terms, 1, 1 + rowSums(exp_terms), "/")  # (n, K)
  proba_red <- cbind(1 - rowSums(proba_red), proba_red)                       # (n, C)
  label_pred <- max.col(proba_red) - 1
  result <- list(pred_proba = proba_red, pred = label_pred)
}

# =====================================================================
# Inference
# =====================================================================
infer_cls <- function(fit, M = 200, alpha = 0.05,
                           parallel = FALSE, n_workers = 4, diag = TRUE) {
  # check arguments
  check_arg_cls_inf(fit, M, alpha, parallel, n_workers, diag)

  caches <- .prepare_inference(fit, diag = diag)
  z <- qnorm(1 - alpha/2)


  one_draw <- function() {
    mu_resample_list <- Map(function(mu, cov) mvrnorm(1, mu = mu, Sigma = cov),
                            fit$mu_list, caches$mu_cov_list)
    rs <- .solve_resample(fit$coef_, caches, mu_resample_list, fit$X0, fit$d, diag)
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
    CI = CI_Union_list
  )
}

# =====================================================================
# Summary
# =====================================================================
summary_cls <- function(fit, infer = NULL, index = NULL, class_index = NULL) {
  width = 8; per_row_coef = 10; per_row_ci = 5; digits_coef = 4; digits_ci = 3
  # ---- basic checks ----
  if (is.null(fit$coef_) || is.null(fit$weight_)) {
    cat("Model is not fitted yet. Run cgdro_() first.\n")
    return(invisible(NULL))
  }
  d <- fit$d; K <- fit$K; num_class <- fit$num_class
  stopifnot(is.numeric(d), is.numeric(K), is.numeric(num_class))

  # ---- helpers ----
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

  # ---- header ----
  cat("Model Summary\n")
  cat("=================================\n")

  # ---- weights ----
  cat("CGDRO Aggregated Weights:\n\n")
  w <- as.numeric(fit$weight_)
  group_idx0 <- seq_along(w) - 1L
  print_chunks("weight_", group_idx0, w, width = width, per_row = 10, fmt = sprintf("%%%d.%df", width, digits_coef), header_label = "group")
  cat("\n")

  # ---- coefficients ----
  cat("=================================\n")
  cat("CGDRO Aggregated Estimators:\n\n")
  coef_mat <- matrix(fit$coef_, nrow = d, byrow = FALSE)
  dim_idx0 <- normalize_indices(index, 1L, d, "index")

  if (is.null(class_index)) {
    class_list <- seq.int(1L, num_class-1)
  } else {
    class_list <- as.integer(class_index)
    if (any(class_list < 1L | class_list > num_class-1))
      stop(sprintf("class_index out of range: must be in [1,%d]", num_class-1))
  }

  for (c_lab in class_list) {
    j <- c_lab - 1L                     # 0..K-1
    vals <- coef_mat[dim_idx0 + 1L, j + 1L]
    cat(sprintf("Class %d coefficients:\n", c_lab))
    print_chunks("coef_", dim_idx0, vals, width = width, per_row = per_row_coef,
                 fmt = sprintf("%%%d.%df", width, digits_coef))
    cat("\n")
  }

  # ---- confidence intervals (optional) ----
  if (!is.null(infer) && !is.null(infer$CI)) {
    cat("=================================\n")
    cat("Confidence Intervals:\n\n")
    # infer$CI is a list of length K; each element is a d x 2 matrix (lb, ub)
    stopifnot(length(infer$CI) == K)
    for (c_lab in class_list) {
      j <- c_lab - 1L
      ci_j <- infer$CI[[j + 1L]]          # d x 2
      ci_sub <- ci_j[dim_idx0 + 1L, , drop = FALSE]
      ci_str <- paste0(
        "(", formatC(ci_sub[, 1], format = "f", digits = digits_ci), ",",
        formatC(ci_sub[, 2], format = "f", digits = digits_ci), ")"
      )
      cat(sprintf("Class %d Confidence Intervals:\n", c_lab))
      print_chunks("CIs", dim_idx0, ci_str, width = max(width, 14), per_row = per_row_ci, fmt = "%s")
      cat("\n")
    }
  } else {
    cat("Confidence Intervals not provided. Run infer() and pass its result via infer=.\n")
  }

  invisible(NULL)
}

`%||%` <- function(a, b) if (!is.null(a)) a else b
