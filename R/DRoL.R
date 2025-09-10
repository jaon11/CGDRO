######################################################################
################################ DRoL ################################
######################################################################
# =====================================================================
# Fit (Closed-form Solution)
# =====================================================================
#' @param seed splitting seed in outcome/density estimation (default: 123)
fit_drol <- function(X_list, y_list, X0 = NULL,
                     f_learner = "xgb",
                     w_learner = "xgb",
                     seed = 123) {
  set.seed(seed)

  # center each source; keep y numeric
  X_list <- lapply(X_list, function(Xi) { Xi <- as.matrix(Xi)})
  y_list <- lapply(y_list, function(yi) as.numeric(yi))

  L <- length(X_list)
  stopifnot(L == length(y_list))
  nls <- vapply(X_list, nrow, 1L)

  # target features
  if (is.null(X0)) X0 <- do.call(rbind, X_list)
  X0 <- as.matrix(X0)
  N <- nrow(X0)

  d <- ncol(X0)


  # check arguments
  check_arg_drol_fit(X_list, y_list, X0,
                     f_learner, w_learner, seed)

  # -------- Plug-in Γ (using full-source outcome models) --------
  source_full_models <- vector("list", L)
  pred_full_mat <- matrix(0, N, L)

  for (l in seq_len(L)) {
    om <- .learn_f(mode = "reg", learner = f_learner)
    om$fit(X_list[[l]], y_list[[l]])
    source_full_models[[l]] <- om
    pred_full_mat[, l] <- om$predict(X0)
  }
  Gamma_plug <- crossprod(pred_full_mat) / N  # t(P) %*% P / N

  # -------- Sample-split models (A/B) for bias-correction --------
  source_A_models <- vector("list", L)
  source_B_models <- vector("list", L)
  density_A_models <- vector("list", L)
  density_B_models <- vector("list", L)
  indA_list <- vector("list", L)
  indB_list <- vector("list", L)

  for (l in seq_len(L)) {
    split_info <- .split_data(X_list[[l]], y_list[[l]], seed = seed)

    sA_model <- .learn_f(mode = "reg", learner = f_learner)
    sB_model <- .learn_f(mode = "reg", learner = f_learner)
    dA_model <- .learn_w(learner = w_learner)
    dB_model <- .learn_w(learner = w_learner)
    indA <- split_info$idx_A; indB <- split_info$idx_B

    # fit models
    sA_model$fit(X_list[[l]][indA, , drop = FALSE], y_list[[l]][indA])
    sB_model$fit(X_list[[l]][indB, , drop = FALSE], y_list[[l]][indB])
    dA_model$fit(X_list[[l]][indA, , drop = FALSE], X0)
    dB_model$fit(X_list[[l]][indB, , drop = FALSE], X0)

    source_A_models[[l]] <- sA_model
    source_B_models[[l]] <- sB_model
    density_A_models[[l]] <- dA_model
    density_B_models[[l]] <- dB_model

    indA_list[[l]] <- indA
    indB_list[[l]] <- indB
  }

  # -------- Bias-corrected Γ --------
  Gamma_corr <- Gamma_plug
  for (k in seq_len(L)) {
    fkA <- source_A_models[[k]]
    fkB <- source_B_models[[k]]
    wkA <- density_A_models[[k]]
    wkB <- density_B_models[[k]]


    for (l in seq_len(L)) {
      flA <- source_A_models[[l]]
      flB <- source_B_models[[l]]
      wlA <- density_A_models[[l]]
      wlB <- density_B_models[[l]]

      num1A <- .bias_correct_term(fkA, flA, wlA, X_list[[l]][indB_list[[l]], , drop = FALSE], y_list[[l]][indB_list[[l]]])
      num2A <- .bias_correct_term(flA, fkA, wkA, X_list[[k]][indB_list[[k]], , drop = FALSE], y_list[[k]][indB_list[[k]]])
      num1B <- .bias_correct_term(fkB, flB, wlB, X_list[[l]][indA_list[[l]], , drop = FALSE], y_list[[l]][indA_list[[l]]])
      num2B <- .bias_correct_term(flB, fkB, wkB, X_list[[k]][indA_list[[k]], , drop = FALSE], y_list[[k]][indA_list[[k]]])

      Gamma_corr[k, l] <- Gamma_corr[k, l] - (num1A + num2A + num1B + num2B) / 2
    }
  }
  Gamma_corr <- (t(Gamma_corr) + Gamma_corr) / 2  # symmetrize

  list(
    # meta
    f_learner = f_learner,
    w_learner = w_learner,
    L = L,
    d = d,
    # data cache
    X0 = X0,
    # predictions + estimators
    pred_full_mat = pred_full_mat,
    Gamma_plug = Gamma_plug,
    Gamma_corr = Gamma_corr,
    family = "drol"
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
  # check arguments
  check_arg_drol_pred(fit, bias_correct, priors, ridge, solver)


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
