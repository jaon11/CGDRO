######################################################################
################################ DRoL ################################
######################################################################
# =====================================================================
# Fit (Closed-form Solution for DRoL)
# =====================================================================
fit_reg_ml <- function(X_list, y_list, X0 = NULL, loss_type = c("reward", "squaredloss", "regret"),
                     f_learner = "xgb", w_learner = "xgb",
                     bias_correct = TRUE, priors = NULL,
                     ridge = 1e-8, solver = c("ECOS", "SCS"),
                     seed = 123) {


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
  check_arg_reg_ml_fit(X_list, y_list, X0, loss_type,
                     f_learner, w_learner,
                     bias_correct, priors,
                     ridge, solver,
                     seed)

  set.seed(seed)

  # -------- Plug-in Γ (using full-source outcome models) --------
  source_full_models <- vector("list", L)
  pred_full_mat <- matrix(0, N, L)
  dev_vec <- numeric(L)

  for (l in seq_len(L)) {
    om <- .learn_f(mode = "reg", learner = f_learner)
    om$fit(X_list[[l]], y_list[[l]])
    source_full_models[[l]] <- om
    pred_full_mat[, l] <- om$predict(X0)
    pred_l <- om$predict(X_list[[l]])
    dev_vec[l] <- sum((y_list[[l]] - pred_l)^2)/(nls[l] - d)  # use n_l - 1 for unbiased est
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



  # -------- Solve for weights (robust, with DCPError message & fallback) --------
  if (!requireNamespace("CVXR", quietly = TRUE)) {
    stop("Package 'CVXR' is required. Please install.packages('CVXR').")
  }

  solver <- match.arg(solver)
  Gamma <- if (bias_correct) Gamma_corr else Gamma_plug
  q <- CVXR::Variable(L)
  G <- Gamma + ridge * diag(L)

  constraints <- list(CVXR::sum_entries(q) == 1, q >= 0)
  if (!is.null(priors)) {
    q0  <- as.numeric(priors[[1]])
    rho <- as.numeric(priors[[2]])
    if (length(q0) != L) stop("prior_weight length must equal number of sources L.")
    if (rho < 0) stop("rho must be nonnegative.")
    constraints <- c(constraints, list(CVXR::p_norm(q - q0, 2) <= rho))
  }

  if (loss_type == "reward") obj <- CVXR::Minimize(CVXR::quad_form(q, G))
  if (loss_type == "squaredloss") obj <- CVXR::Minimize(CVXR::quad_form(q, G) - t(q) %*% (diag(G) + dev_vec))
  if (loss_type == "regret") obj <- CVXR::Minimize(CVXR::quad_form(q, G) - t(q) %*% (diag(G)))

  # Wrap problem build + solve in tryCatch
  q_opt <- tryCatch({
    # Problem construction (can throw DCP errors if mis-specified)
    prob <- tryCatch(
      CVXR::Problem(obj, constraints),
      error = function(e) {
        if (grepl("DCP|DGP|DPP|disciplined", conditionMessage(e), ignore.case = TRUE)) {
          message(" f_learner or w_learner is not well-fitted.")
        }
        stop(e)
      }
    )

    # First attempt with user-specified solver
    res <- try(CVXR::solve(prob, solver = solver), silent = TRUE)
    if (inherits(res, "try-error") || isTRUE(res$status %in% c("infeasible", "unbounded", "solver_error"))) {
      # Fallback to SCS
      res <- try(CVXR::solve(prob, solver = "SCS"), silent = TRUE)
    }

    # If still failing, print message and fallback to uniform weights
    if (inherits(res, "try-error") || isTRUE(res$status %in% c("infeasible", "unbounded", "solver_error"))) {
      message(" f_learner or w_learner is not well-fitted.")
      qo <- rep(1 / L, L)
      return(qo)
    }

    # Extract and sanitize solution
    qo <- as.numeric(res$getValue(q))
    qo[is.na(qo)] <- 0
    qo[qo < 0] <- 0
    s <- sum(qo)
    if (s <= 0) {
      message(" f_learner or w_learner is not well-fitted.")
      qo <- rep(1 / L, L)
    } else {
      qo <- qo / s
    }
    qo
  }, error = function(e) {
    # Catch any remaining errors (including DCP at solve-time)
    if (grepl("DCP|DGP|DPP|disciplined", conditionMessage(e), ignore.case = TRUE)) {
      message(" f_learner or w_learner is not well-fitted.")
    }
    # Safe fallback
    rep(1 / L, L)
  })



  list(
    # meta
    f_learner = f_learner,
    w_learner = w_learner,
    L = L,
    d = d,
    # data cache
    X0 = X0,
    dev_vec = dev_vec,
    source_full_models = source_full_models,
    # predictions + estimators
    pred_full_mat = pred_full_mat,
    Gamma_plug = Gamma_plug,
    Gamma_corr = Gamma_corr,
    weight_ = q_opt,
    family = "reg_ml"
  )
}

# =====================================================================
# Predict on target
# =====================================================================
predict_reg_ml <- function(fit, X=NULL) {
  if (is.null(X)) {
    pred_full_mat = fit$pred_full_mat
  } else {
    if (dim(X)[2] != fit$d) {
      stop("Dimension of X does not match the fitted model.")
    }
    X <- as.matrix(X)
    pred_full_mat <- matrix(0, nrow(X), fit$L)
    for (l in seq_len(fit$L)) {
      om <- fit$source_full_models[[l]]
      pred_full_mat[, l] <- om$predict(X)
    }
  }


  pred <- as.numeric(pred_full_mat %*% fit$weight)
  pred
}
