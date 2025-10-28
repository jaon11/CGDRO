######################### Utility Functions Used in CGDRO #################################
###########################################################################################

# =====================================================================
# Dependencies
# =====================================================================
library(CVXR)
library(nnet)
library(xgboost)
library(glmnet)
library(SIHR)
library(MASS)
library(densratio)
# =====================================================================
# Helpers
# =====================================================================

# ================================================================
# Unified learners: learn_f (outcome) and learn_w (density ratio)
# ================================================================
train.fun <- function(X, y, lambda=NULL, intercept=FALSE){
  if(is.null(lambda)) lambda = 'CV.min'
  p = ncol(X)
  htheta <- if (lambda == "CV.min") {
    outLas <- cv.glmnet(X, y, family = "gaussian", alpha = 1,
                        intercept = intercept, standardize = T)
    as.vector(coef(outLas, s = outLas$lambda.min))
  } else if (lambda == "CV") {
    outLas <- cv.glmnet(X, y, family = "gaussian", alpha = 1,
                        intercept = intercept, standardize = T)
    as.vector(coef(outLas, s = outLas$lambda.1se))
  } else {
    outLas <- glmnet(X, y, family = "gaussian", alpha = 1,
                     intercept = intercept, standardize = T)
    as.vector(coef(outLas, s = lambda))
  }
  if(intercept==FALSE) htheta = htheta[2:(p+1)]

  return(list(lasso.est = htheta))
}
# ---- Small utilities ----------------------------------------------------------
.clip01 <- function(p, lo = 1e-8, hi = 1 - 1e-8) pmin(pmax(p, lo), hi)

.mm_if_needed <- function(X) {
  if (is.data.frame(X)) {
    mm <- stats::model.matrix(~ . - 1, data = X); storage.mode(mm) <- "double"; return(mm)
  }
  data.matrix(X)
}

.pull_nrounds <- function(params, default = 200L) {
  if (is.null(params)) return(list(nrounds = default, params_wo = NULL))
  stopifnot(is.list(params))
  nr <- if (!is.null(params$nrounds)) as.integer(params$nrounds) else default
  params$nrounds <- NULL
  list(nrounds = nr, params_wo = params)
}

.ensure_feature_names <- function(X, feat_names) {
  Xdf <- as.data.frame(X, check.names = FALSE)
  if (is.null(colnames(Xdf))) colnames(Xdf) <- paste0("X", seq_len(ncol(Xdf)))
  missing <- setdiff(feat_names, colnames(Xdf))
  if (length(missing)) Xdf[missing] <- 0
  Xdf[, feat_names, drop = FALSE]
}

# ---- XGBoost CV helper (5-fold + early stop) ----------------------------------
.xgb_cv_pick <- function(dtrain, base_params, seed = 123,
                         grid = expand.grid(
                           eta              = c(0.01, 0.05, 0.1),
                           max_depth        = c(3, 6, 9),
                           subsample        = c(0.8, 1.0),
                           colsample_bytree = c(0.8, 1.0),
                           KEEP.OUT.ATTRS   = FALSE,
                           stringsAsFactors = FALSE
                         ),
                         objective_type = c("reg", "bin", "multi"),
                         K = NULL) {
  objective_type <- match.arg(objective_type)
  set.seed(seed)
  best_score <- Inf; best_iter <- 200L; best_row <- NULL
  max_nrounds <- 1000L; early_stop <- 50L

  for (i in seq_len(nrow(grid))) {
    row <- as.list(grid[i, , drop = FALSE])
    params_try <- utils::modifyList(base_params, row)

    cv <- xgboost::xgb.cv(
      params                = params_try,
      data                  = dtrain,
      nrounds               = max_nrounds,
      nfold                 = 5,
      stratified            = (objective_type != "reg"),
      early_stopping_rounds = early_stop,
      verbose               = 0,
      maximize              = FALSE,
      showsd                = TRUE
    )

    bi <- cv$best_iteration
    if (is.null(bi) || is.na(bi) || bi <= 0) bi <- nrow(cv$evaluation_log)

    metric_name <- switch(objective_type,
                          reg   = "test_rmse_mean",
                          bin   = "test_logloss_mean",
                          multi = "test_mlogloss_mean"
    )
    score <- cv$evaluation_log[[metric_name]][bi]

    if (!is.na(score) && score < best_score) {
      best_score <- score; best_iter <- bi; best_row <- row
    }
  }

  if (is.null(best_row)) {
    best_row <- as.list(grid[1, , drop = FALSE]); best_iter <- 200L
  }
  list(params = utils::modifyList(base_params, best_row), nrounds = best_iter)
}

# =====================================================================
# 1) Outcome learner
# =====================================================================
.learn_f <- function(mode = c("reg", "cls"),
                    learner = c("linear", "xgb", "xgb.cv","high_d"),
                    params = NULL,
                    sample_weight = NULL,
                    lambda_val = NULL,
                    seed = 123) {
  mode <- match.arg(mode)
  learner <- match.arg(learner)
  set.seed(seed)

  if (mode == "reg") {
    # ---------------- Regression ----------------
    if (learner == "linear") {
      model <- NULL
      fit <- function(X, y) {
        X <- as.data.frame(X); y <- as.numeric(y)
        if (is.null(sample_weight)) {
          model <<- stats::lm(y ~ ., data = data.frame(y = y, X))
        } else {
          model <<- stats::lm(y ~ ., data = data.frame(y = y, X), weights = sample_weight)
        }
        invisible(NULL)
      }
      predict <- function(Xnew) {
        as.numeric(stats::predict(model, newdata = as.data.frame(Xnew)))
      }
      return(list(fit = fit, predict = predict, model = function() model, mode = "reg", learner = "linear"))
    }

    if (learner %in% c("xgb", "xgb.cv")) {
      if (!requireNamespace("xgboost", quietly = TRUE))
        stop("Need package 'xgboost' for learner = '", learner, "'.")
      model <- NULL; nrounds_used <- NA_integer_; params_used <- NULL

      fit <- function(X, y) {
        Xm <- .mm_if_needed(X); y <- as.numeric(y)
        sw <- if (is.null(sample_weight)) rep(1, nrow(Xm)) else sample_weight
        dtrain <- xgboost::xgb.DMatrix(data = Xm, label = y, weight = sw, missing = NA)

        base <- list(objective = "reg:squarederror", eval_metric = "rmse")
        pr <- .pull_nrounds(params)
        base <- utils::modifyList(base, if (is.null(pr$params_wo)) list() else pr$params_wo)

        if (learner == "xgb.cv") {
          picked <- .xgb_cv_pick(dtrain, base_params = base, seed = seed, objective_type = "reg")
          params_used <<- picked$params; nrounds_used <<- picked$nrounds
        } else {
          params_used <<- base; nrounds_used <<- pr$nrounds
        }

        model <<- xgboost::xgb.train(params = params_used, data = dtrain,
                                     nrounds = nrounds_used, verbose = 0)
        invisible(NULL)
      }
      predict <- function(Xnew) {
        Xn <- .mm_if_needed(Xnew)
        # IMPORTANT: call the generic so predict.xgb.Booster is dispatched
        as.numeric(stats::predict(model, newdata = Xn, missing = NA))
      }
      return(list(fit = fit, predict = predict,
                  model = function() model, mode = "reg",
                  learner = learner, params_used = function() params_used,
                  nrounds = function() nrounds_used))
    }

    # ---------------- High-dimensional Lasso (glmnet) ----------------
    if (learner == "high_d") {
  if (!requireNamespace("glmnet", quietly = TRUE))
    stop("Need package 'glmnet' for learner = 'high_d'.")

  # Stored state
  model <- NULL        # list(beta = numeric p_c, intercept = TRUE/FALSE)
  p_fit <- NULL        # ncol of raw features after .mm_if_needed
  p_fit_c <- NULL      # ncol of constructed design Xc (with or without intercept col)
  used_intercept <- FALSE



  fit <- function(X, y, intercept = FALSE) {
    # Construct the exact design we will learn on
    Xc <- X
    y  <- as.numeric(y)

    # glmnet handles NO internal intercept; our first column (if any) is the intercept column.
    if (is.null(lambda_val) || identical(lambda_val, "CV.min")) {
      cvfit <- glmnet::cv.glmnet(
        x = Xc, y = y, family = "gaussian",
        alpha = 1, intercept = FALSE, standardize = TRUE,
        weights = sample_weight
      )
      beta <- as.numeric(stats::coef(cvfit, s = "lambda.min"))[-1]  # aligned to Xc columns directly
    } else if (identical(lambda_val, "CV")) {
      cvfit <- glmnet::cv.glmnet(
        x = Xc, y = y, family = "gaussian",
        alpha = 1, intercept = FALSE, standardize = TRUE,
        weights = sample_weight
      )
      lam  <- cvfit$lambda.1se
      beta <- as.numeric(stats::coef(cvfit, s = lam))[-1]
    } else {
      fit0 <- glmnet::glmnet(
        x = Xc, y = y, family = "gaussian",
        alpha = 1, lambda = lambda_val,
        intercept = FALSE, standardize = TRUE,
        weights = sample_weight
      )
      s_use <- if (length(lambda_val) == 1) lambda_val else lambda_val[1]
      beta  <- as.numeric(stats::coef(fit0, s = s_use))[-1]
    }

    # Persist state
    model <<- list(beta = beta, intercept = intercept)
    p_fit <<- ncol(.mm_if_needed(X))
    p_fit_c <<- ncol(Xc)                  # this is the length of beta
    used_intercept <<- intercept
    invisible(NULL)
  }

  predict <- function(Xnew) {
    if (is.null(model)) stop("Call $fit(X, y, intercept=...) before predicting.")
    Xc_new <- Xnew
    if (ncol(Xc_new) != p_fit_c) {
      stop(sprintf("Column mismatch: trained with p_c=%d, got p_c=%d.", p_fit_c, ncol(Xc_new)))
    }
    as.numeric(Xc_new %*% model$beta)
  }

  return(list(
    fit = fit,
    predict = predict,
    # coef() returns EXACT beta_init compatible with Xc %*% beta_init
    coef = function() {
      if (is.null(model)) stop("Model not fit.")
      model$beta
    },
    # convenience getters
    model = function() model,
    mode = "reg",
    p_fit = function() p_fit,         # raw feature count after .mm_if_needed
    p_fit_c = function() p_fit_c,     # design-column count (matches length of coef)
    intercept_used = function() used_intercept
  ))
}


  } else {
    # ---------------- Classification ----------------
    if (learner == "linear") {
      if (!requireNamespace("nnet", quietly = TRUE))
        stop("Need package 'nnet' for multinomial classification.")

      model <- NULL
      feat_names <- NULL
      classes <- NULL

      fit <- function(X, y) {
        # keep names stable
        Xdf <- as.data.frame(X, check.names = FALSE)
        if (is.null(colnames(Xdf))) colnames(Xdf) <- paste0("X", seq_len(ncol(Xdf)))
        feat_names <<- colnames(Xdf)
        classes <<- levels(as.factor(y))

        df <- data.frame(y = as.factor(y), Xdf, check.names = FALSE)
        args <- c(list(formula = y ~ ., data = df, trace = FALSE, MaxNWts = 100000), params)
        model <<- do.call(nnet::multinom, args)
        invisible(NULL)
      }

      predict <- function(Xnew) {
        Xdf <- .ensure_feature_names(Xnew, feat_names)
        as.character(stats::predict(model, newdata = Xdf, type = "class"))
      }

      predict_proba <- function(Xnew) {
        Xdf <- .ensure_feature_names(Xnew, feat_names)
        p <- stats::predict(model, newdata = Xdf, type = "probs")  # matrix or data.frame
        p <- as.matrix(.clip01(p, 1e-6, 1 - 1e-6))

        # ensure consistent column order (important when levels() order differs)
        if (!is.null(classes)) {
          # add any missing class columns with ~0 prob (rare but safe)
          miss <- setdiff(classes, colnames(p))
          if (length(miss)) {
            for (m in miss) p <- cbind(p, setNames(matrix(0, nrow(p), 1), m))
          }
          p <- p[, classes, drop = FALSE]
        }
        p
      }

      return(list(
        fit = fit,
        predict = predict,
        predict_proba = predict_proba,
        model = function() model,
        classes = function() classes,
        feature_names = function() feat_names,
        mode = "cls",
        learner = "linear"
      ))
    }


    if (learner %in% c("xgb", "xgb.cv")) {
      if (!requireNamespace("xgboost", quietly = TRUE))
        stop("Need package 'xgboost' for learner = '", learner, "'.")
      model <- NULL; classes <- NULL; params_used <- NULL; nrounds_used <- NA_integer_

      fit <- function(X, y) {
        y_fac <- as.factor(y); classes <<- levels(y_fac)
        y_int <- as.integer(y_fac) - 1L
        Xm <- .mm_if_needed(X)
        sw <- if (is.null(sample_weight)) NULL else sample_weight
        dtrain <- xgboost::xgb.DMatrix(data = Xm, label = y_int, weight = sw, missing = NA)

        base <- list(objective = "multi:softprob", eval_metric = "mlogloss", num_class = length(classes))
        pr <- .pull_nrounds(params)
        base <- utils::modifyList(base, if (is.null(pr$params_wo)) list() else pr$params_wo)

        if (learner == "xgb.cv") {
          picked <- .xgb_cv_pick(dtrain, base_params = base, seed = seed,
                                 objective_type = "multi", K = length(classes))
          params_used <<- picked$params; nrounds_used <<- picked$nrounds
        } else {
          params_used <<- base; nrounds_used <<- pr$nrounds
        }

        model <<- xgboost::xgb.train(params = params_used, data = dtrain,
                                     nrounds = nrounds_used, verbose = 0)
        invisible(NULL)
      }

      predict_proba <- function(Xnew) {
        Xn  <- .mm_if_needed(Xnew)
        raw <- stats::predict(model, newdata = Xn, missing = NA)  # generic -> xgb method
        K <- length(classes)
        probs <- matrix(raw, ncol = K, byrow = TRUE)
        colnames(probs) <- classes
        .clip01(probs, 1e-6, 1 - 1e-6)
      }
      predict <- function(Xnew) {
        pr <- predict_proba(Xnew)
        classes[max.col(pr, ties.method = "first")]
      }

      return(list(fit = fit, predict = predict, predict_proba = predict_proba,
                  model = function() model, classes = function() classes,
                  mode = "cls", learner = learner, params_used = function() params_used,
                  nrounds = function() nrounds_used))
    }
  }

  stop("Unsupported combination: mode=", mode, ", learner=", learner)
}

# =====================================================================
# 2) Density-ratio learner  w(x) = p0(x)/p(x)
# =====================================================================
.learn_w <- function(learner = c("linear", "xgb", "xgb.cv", "ulsif"),
                     params = NULL,
                     seed = 123) {
  learner <- match.arg(learner)
  set.seed(seed)

  model <- NULL
  ratio <- NA_real_
  params_used <- NULL
  nrounds_used <- NA_integer_

  # ---------- helpers ----------
  as_matrix2d <- function(M) {
    M <- .mm_if_needed(M)
    if (is.null(dim(M))) M <- matrix(M, ncol = 1L)
    storage.mode(M) <- "double"
    M
  }

  # ---------- fit ----------
  fit <- function(X, X_target) {
    X  <- as_matrix2d(X)
    X0 <- as_matrix2d(X_target)

    if (learner == "ulsif") {
      if (!requireNamespace("densratio", quietly = TRUE))
        stop("Need package 'densratio' for learn_w(learner = 'ulsif').")

      ratio <<- 1.0  # ulsif directly estimates p0/p

      # Optional args
      sigma_arg <- if (!is.null(params) && !is.null(params$sigma)) params$sigma else "auto"
      fold_arg  <- if (!is.null(params) && !is.null(params$fold))  as.integer(params$fold) else 5L

      # densratio::ulsif expects numerator = target (X0), denominator = source (X)
      kl <- try(densratio::densratio(x1 = X0, x2 = X, sigma = sigma_arg, fold = fold_arg, verbose = FALSE),
                silent = TRUE)

      if (!inherits(kl, "try-error")) {
        # Store a simple adapter so predict() can always call model$predict_fn(...)
        model <<- list(
          method = "ulsif_densratio",
          predict_fn = function(Xnew) {
            Xn <- as_matrix2d(Xnew)
            w  <- as.numeric(kl$compute_density_ratio(Xn))
            pmin(pmax(w, 1e-3), 1e3)
          }
        )
        params_used <<- list(method = "uLSIF", backend = "densratio",
                             sigma = tryCatch(kl$sigma, error = function(e) NA_real_),
                             fold = fold_arg)
        nrounds_used <<- NA_integer_
        return(invisible(NULL))
      }


      # Last-resort stub (uniform weights at predict time)
      warning("uLSIF fitting failed in densratio; falling back to uniform weights.")
      model <<- list(
        method = "uLSIF_uniform_stub",
        predict_fn = function(Xnew) rep(1, nrow(as_matrix2d(Xnew)))
      )
      params_used <<- list(method = "uLSIF", backend = "stub")
      nrounds_used <<- NA_integer_
      return(invisible(NULL))
    }

    # -------- existing learners --------
    ratio <<- nrow(X) / nrow(X0)

    Xc <- rbind(X, X0)
    yc <- c(rep(0, nrow(X)), rep(1, nrow(X0)))

    if (learner == "linear") {
      if (!requireNamespace("glmnet", quietly = TRUE))
        stop("Need package 'glmnet' for learn_w(learner='linear').")
      cvfit <- glmnet::cv.glmnet(Xc, yc, family = "binomial", type.measure = "deviance")
      model <<- cvfit
      params_used <<- list(lambda = "lambda.min")
      return(invisible(NULL))
    }

    if (!requireNamespace("xgboost", quietly = TRUE))
      stop("Need package 'xgboost' for learn_w(learner='", learner, "').")

    base <- list(objective = "binary:logistic", eval_metric = "logloss")
    pr <- .pull_nrounds(params)
    base <- utils::modifyList(base, if (is.null(pr$params_wo)) list() else pr$params_wo)

    dtrain <- xgboost::xgb.DMatrix(data = Xc, label = yc, missing = NA)

    if (learner == "xgb.cv") {
      picked <- .xgb_cv_pick(dtrain, base_params = base, seed = seed, objective_type = "bin")
      params_used <<- picked$params; nrounds_used <<- picked$nrounds
    } else {
      params_used <<- base; nrounds_used <<- pr$nrounds
    }

    model <<- xgboost::xgb.train(params = params_used, data = dtrain,
                                 nrounds = nrounds_used, verbose = 0)
    invisible(NULL)
  }

  # ---------- predict ----------
  predict <- function(Xnew) {
    if (is.null(model)) stop("Call $fit(X, X_target) before predicting ratios.")

    # If the model exposes a predict_fn (uLSIF adapters), just use it
    if (is.list(model) && !is.null(model$predict_fn)) {
      return(model$predict_fn(Xnew))
    }

    Xn <- as_matrix2d(Xnew)

    if (inherits(model, "cv.glmnet")) {
      p1 <- as.numeric(stats::predict(model, Xn, s = "lambda.min", type = "response"))
    } else if (inherits(model, "xgb.Booster")) {
      p1 <- as.numeric(stats::predict(model, newdata = Xn, missing = NA))
    } else {
      stop("Unknown model class in .learn_w()$model.")
    }

    p1 <- .clip01(p1, 1e-8, 1 - 1e-8)
    w  <- (p1 / (1 - p1)) * ratio
    pmin(pmax(w, 1e-3), 1e3)
  }

  list(fit = fit, predict = predict,
       model = function() model,
       learner = learner,
       params_used = function() params_used,
       nrounds = function() nrounds_used)
}

# =====================================================================
# 3) Data splitting utility
.split_data <- function(X, y = NULL, train_frac = 0.5, seed = 123) {
  set.seed(seed)
  #n <- nrow(X)
  #cat("Splitting data: n =", n, ", size of A =", floor(train_frac * nrow(X)), "train_frac =", train_frac, "\n")
  idx <- sample(seq_len(nrow(X)), size = floor(train_frac * nrow(X)))

  A <- list(
    X = X[idx, , drop = FALSE],
    y = if (!is.null(y)) y[idx] else NULL
  )
  B <- list(
    X = X[-idx, , drop = FALSE],
    y = if (!is.null(y)) y[-idx] else NULL
  )

  list(A = A, B = B, idx_A = idx, idx_B = setdiff(seq_len(nrow(X)), idx))
}

######################################################################
######################### DRlm-Classification ########################
######################################################################
#### xgb.CV of density ratio estimation: get faster; default xgb did not work.
.softmax_reduced <- function(x) {
  # x: (n, K) where K = C-1 (reference class not included)
  # returns (n, K), probs for non-reference classes
  x_max <- apply(x, 1, function(r) max(c(0, r)))
  exp_x <- exp(sweep(x, 1, x_max, "-"))
  denom <- 1 + rowSums(exp_x)
  sweep(exp_x, 1, denom, "/")
}

# ---- Doubly-robust mu ----
.compute_mu_list <- function(X_list, y_list, X0, probaX_list, probaX0_list, omegaX_list, num_class, d) {
  L <- length(X_list)
  mu_list <- vector("list", L)
  for (l in seq_len(L)) {
    n_l <- nrow(X_list[[l]]); n0 <- nrow(X0)
    y_onehot <- diag(num_class)[as.integer(as.factor(y_list[[l]])), , drop = FALSE]  # 1..C
    # term1 uses non-reference classes: drop the first column (class 1)
    term1 <- - t(X0) %*% (probaX0_list[[l]][, -1, drop = FALSE]) / n0
    term2 <- - t(X_list[[l]] * omegaX_list[[l]]) %*%
      (y_onehot[, -1, drop = FALSE] - probaX_list[[l]][, -1, drop = FALSE]) / n_l
    mu_list[[l]] <- as.numeric(term1 + term2)  # flatten col-major
  }
  mu_list
}

# ---- Objective pieces ----
.compute_primal <- function(theta, mu_list, X0, d) {
  g_theta <- max(sapply(mu_list, function(mu) sum(theta * mu)))
  K <- length(theta) / d
  theta_mat <- matrix(theta, nrow = d, byrow = FALSE)
  logits <- X0 %*% theta_mat
  logits_max <- apply(logits, 1, max)
  stable <- sweep(logits, 1, logits_max, "-")
  exp_terms <- exp(stable)
  S_theta <- mean(logits_max + log(exp(-logits_max) + rowSums(exp_terms)))
  g_theta + S_theta
}

.compute_grad <- function(theta, gamma, mu_list, X0, d) {
  K <- length(theta) / d
  theta_mat <- matrix(theta, nrow = d, byrow = FALSE)
  proba_mat <- .softmax_reduced(X0 %*% theta_mat)  # (n0, K)
  grad_S <- as.numeric(t(X0) %*% proba_mat / nrow(X0))
  grad_theta <- grad_S + Reduce(`+`, Map(function(g, mu) g * mu, gamma, mu_list))
  grad_gamma <- vapply(mu_list, function(mu) sum(theta * mu), numeric(1))
  list(grad_theta = grad_theta, grad_gamma = grad_gamma)
}

.compute_dual_value <- function(theta_init, gamma, mu_list, X0, d) {
  f <- function(theta) {
    obj <- sum(gamma * vapply(mu_list, function(mu) sum(theta * mu), numeric(1)))
    theta_mat <- matrix(theta, nrow = d, byrow = FALSE)
    logits <- X0 %*% theta_mat
    logits_max <- apply(logits, 1, max)
    stable <- sweep(logits, 1, logits_max, "-")
    exp_terms <- exp(stable)
    obj + mean(logits_max + log(exp(-logits_max) + rowSums(exp_terms)))
  }
  optim(theta_init, f, method = "L-BFGS-B")$value
}


# ---- Prepara for Inference ----
.prepare_inference <- function(fit, diag = TRUE) {
  n0 <- nrow(fit$X0); d <- fit$d; K <- fit$K
  theta_mat <- matrix(fit$coef_, nrow = d, byrow = FALSE)
  proba_mat <- .softmax_reduced(fit$X0 %*% theta_mat)

  # Hessian
  H <- matrix(0, d*K, d*K)
  for (j in seq_len(K)) {
    for (k in seq_len(K)) {
      pj <- proba_mat[, j]; pk <- proba_mat[, k]
      w <- pj * ((j == k) - pk)
      H_block <- t(fit$X0) %*% (diag(w, n0, n0) %*% fit$X0) / n0
      rows <- ((j-1)*d + 1):(j*d); cols <- ((k-1)*d + 1):(k*d)
      H[rows, cols] <- H_block
    }
  }
  H_inv <- if (diag) diag(1/diag(H)) else solve(H)

  gradS <- as.numeric(t(fit$X0) %*% proba_mat / n0)

  diag_cov <- function(psi) diag(colMeans(psi^2) - colMeans(psi)^2)
  psiS <- t(vapply(seq_len(n0), function(i) kronecker(proba_mat[i, ], fit$X0[i, ]), numeric(d*K)))
  gradS_cov <- (if (diag) diag_cov(psiS) else stats::cov(psiS)) / n0

  mu_cov_list <- list(); mu_gradS_cov_list <- list()
  for (l in seq_len(fit$L)) {
    n_l <- nrow(fit$X_list[[l]])
    y_onehot <- diag(fit$num_class)[as.integer(as.factor(fit$y_list[[l]])), , drop = FALSE]
    psi1 <- t(vapply(seq_len(n0), function(i) -kronecker(fit$probaX0_list[[l]][i, -1], fit$X0[i, ]), numeric(d*K)))
    cov1 <- (if (diag) diag_cov(psi1) else stats::cov(psi1)) / n0
    psi2 <- t(vapply(seq_len(n_l), function(i) -kronecker(y_onehot[i, -1] - fit$probaX_list[[l]][i, -1],
                                                          fit$X_list[[l]][i, ] * fit$omegaX_list[[l]][i]), numeric(d*K)))
    cov2 <- (if (diag) diag_cov(psi2) else stats::cov(psi2)) / n_l
    mu_cov_list[[l]] <- cov1 + cov2
    if (diag) {
      cov_diag <- colMeans(psi1 * psiS) - colMeans(psi1) * colMeans(psiS)
      mu_gradS_cov_list[[l]] <- diag(cov_diag) / n0
    } else {
      mu_gradS_cov_list[[l]] <- stats::cov(cbind(psi1, psiS))[1:(d*K), (d*K + 1):(2*d*K)] / n0
    }
  }

  list(H_inv = H_inv, gradS = gradS, gradS_cov = gradS_cov,
       mu_cov_list = mu_cov_list, mu_gradS_cov_list = mu_gradS_cov_list)
}

.solve_resample <- function(theta, caches, mu_resample_list, X0, d, diag) {
  H_inv <- caches$H_inv
  H_inv_diag <- if (diag && length(dim(H_inv)) == 2) diag(H_inv) else NULL
  softmax_vec <- function(u) { u <- u - max(u); e <- exp(u); e/sum(e) }

  obj_u <- function(u) {
    gamma <- softmax_vec(u)
    weighted_mu <- Reduce(`+`, Map(function(g, mu) g * mu, gamma, mu_resample_list))
    g <- weighted_mu + caches$gradS
    quad <- if (!is.null(H_inv_diag)) 0.5 * sum((g^2) * H_inv_diag) else 0.5 * drop(t(g) %*% H_inv %*% g)
    lin  <- - sum(gamma * vapply(mu_resample_list, function(mu) sum(mu * theta), numeric(1)))
    quad - lin
  }
  gr_u <- function(u) {
    gamma <- softmax_vec(u)
    weighted_mu <- Reduce(`+`, Map(function(g, mu) g * mu, gamma, mu_resample_list))
    g <- weighted_mu + caches$gradS
    Hg <- if (!is.null(H_inv_diag)) g * H_inv_diag else as.numeric(H_inv %*% g)
    grad_gamma <- vapply(mu_resample_list, function(mu) sum(Hg * mu) - sum(mu * theta), numeric(1))
    J <- diag(gamma) - outer(gamma, gamma)
    as.numeric(J %*% grad_gamma)
  }
  u0 <- rep(0, length(mu_resample_list))
  optg <- optim(u0, obj_u, gr_u, method = "BFGS")
  gamma_resample <- softmax_vec(optg$par)

  weighted_mu_sum <- Reduce(`+`, Map(function(g, mu) g * mu, gamma_resample, mu_resample_list))

  obj_theta <- function(th) {
    th_mat <- matrix(th, nrow = d, byrow = FALSE)
    Xth <- X0 %*% th_mat
    mv <- pmax(0, apply(Xth, 1, max))
    safe_exp <- exp(sweep(Xth, 1, mv, "-"))
    log_term <- mean(mv + log(1 + rowSums(safe_exp)))
    lin_term <- sum(th * weighted_mu_sum)
    lin_term + log_term
  }
  grad_theta <- function(th) {
    th_mat <- matrix(th, nrow = d, byrow = FALSE)
    X <- X0
    Xth <- X %*% th_mat
    mv <- apply(Xth, 1, max)
    exp_th <- exp(sweep(Xth, 1, mv, "-"))
    probs <- sweep(exp_th, 1, 1 + rowSums(exp_th), "/")
    log_grad <- as.numeric(t(X) %*% probs)/nrow(X)
    log_grad + weighted_mu_sum
  }
  optt <- optim(theta, obj_theta, grad_theta, method = "L-BFGS-B")
  list(theta = optt$par, gamma = gamma_resample)
}

.compute_variance_resample <- function(gamma_resample, caches) {
  term1 <- Reduce(`+`, Map(function(g, cov) (g^2) * cov, gamma_resample, caches$mu_cov_list))
  term2 <- caches$gradS_cov
  term3 <- Reduce(`+`, Map(function(g, cov) g * cov, gamma_resample, caches$mu_gradS_cov_list))
  W <- term1 + term2 - 2 * term3
  caches$H_inv %*% W %*% caches$H_inv
}



######################################################################
########################### DRlm-Regression ##########################
######################################################################


.compute_dev <- function(Xb, y, sparsity = 0) {
  y <- as.numeric(y); r <- y - as.numeric(Xb)
  n <- length(y)
  denom <- max(0.7 * n, n - sparsity)
  sum(r * r) / denom
}


.opt_weight<-function(Gamma, delta=0, loss_type='reward', dev_vec=NULL){
  ## Purpose: Compute Ridge-type weight vector
  ## Returns: weight:  the minimizer \eqn{\gamma}
  ##          reward:  the value of penalized reward
  ## ----------------------------------------------------
  ## Arguments: Gamma: regression covariance matrix, of dimension \eqn{L} x \eqn{L}
  ##            delta the ridge penalty level, non-positive.
  ##            loss_type the type of objective function, either 'reward', 'squaredloss', or 'regret' (Default = 'reward')
  ##            report.reward the reward is computed or not (Default = `TRUE`)
  ## ----------------------------------------------------

  L<-dim(Gamma)[2]
  opt.weight<-rep(NA, L)
  opt.reward<-NA
  # Problem definition
  v<-Variable(L)
  Diag.matrix<-diag(eigen(Gamma)$values)
  for(ind in 1:L){
    Diag.matrix[ind,ind]<-max(Diag.matrix[ind,ind],0.001)
  }
  Gamma_positive<-eigen(Gamma)$vectors%*%Diag.matrix%*%t(eigen(Gamma)$vectors)
  Gamma_diag<-(diag(Gamma))
  if (loss_type == "reward") {
    objective <- Minimize(quad_form(v, Gamma_positive + diag(delta, L)))
  } else if (loss_type == "squaredloss") {
    objective <- Minimize(
      quad_form(v, Gamma_positive + diag(delta, L)) -
        t(v) %*% (Gamma_diag + dev_vec)
    )
  } else if (loss_type == "regret") {
    objective <- Minimize(
      quad_form(v, Gamma_positive + diag(delta, L)) -
        t(v) %*% Gamma_diag
    )
  } else {
    stop("Unsupported loss_type: ", loss_type)
  }
  constraints <- list(v >= 0, sum(v)== 1)
  prob.weight<- Problem(objective, constraints)
  if(is_dcp(prob.weight)){
    result<- solve(prob.weight)
    opt.status<-result$status
    opt.sol<-result$getValue(v)
    for(l in 1:L){
      opt.weight[l]<-opt.sol[l]*(abs(opt.sol[l])>10^{-8})
    }
  }

 opt.weight<-opt.weight/sum(opt.weight)
 opt.weight
}
.index_map <- function(L, l, k) {
  # returns 1-based index for vectorized lower-triangle (column-wise), with l>=k
  l0 <- l - 1; k0 <- k - 1
  as.integer((2 * L - (k0 + 1)) * k0 / 2 + l0 + 1)
}

.gensamples <- function(gen_mu, gen_Cov, gen_size = 500,
                        threshold = 0, alpha_thres = 0.01) {
  gen_mu <- as.numeric(gen_mu)
  gen_dim <- length(gen_mu)
  out <- matrix(0, gen_size, gen_dim)

  if (threshold == 2) {
    return(MASS::mvrnorm(gen_size, mu = gen_mu, Sigma = gen_Cov))
  }

  n_picked <- 0
  if (threshold == 0) {
    thres <- stats::qnorm(1 - alpha_thres / (2 * gen_dim))
    sdv <- sqrt(pmax(diag(gen_Cov), 1e-12))
    while (n_picked < gen_size) {
      S <- as.numeric(MASS::mvrnorm(1, mu = rep(0, gen_dim), Sigma = gen_Cov))
      if (max(abs(S / sdv)) <= thres) {
        n_picked <- n_picked + 1
        out[n_picked, ] <- gen_mu + S
      }
    }
  } else if (threshold == 1) {
    ed <- eigen((gen_Cov + t(gen_Cov)) / 2, symmetric = TRUE)
    A <- ed$vectors %*% diag(sqrt(pmax(ed$values, 1e-12))) %*% t(ed$vectors)
    thres <- stats::qchisq(1 - alpha_thres, df = gen_dim)
    while (n_picked < gen_size) {
      z <- rnorm(gen_dim)
      if (sum(z^2) <= thres) {
        n_picked <- n_picked + 1
        out[n_picked, ] <- gen_mu + as.numeric(A %*% z)
      }
    }
  } else {
    stop("threshold must be 0, 1, or 2.")
  }
  out
}

.compute_Var_Gamma_ld <- function(fit, tau = 0.2, ridge = 1e-8) {
  L  <- fit$L
  p  <- fit$d                     # this is d (+1 if intercept=TRUE) from fit_reg_ld
  use_intercept <- isTRUE(fit$intercept)

  gen_dim   <- L * (L + 1) / 2
  Var_Gamma <- matrix(NA_real_, gen_dim, gen_dim)

  # sizes
  ns <- vapply(fit$X_list, nrow, integer(1))

  # Target design already prepared by fit_reg_ld (with/without intercept)
  X0_use <- fit$X0_use
  N0     <- nrow(X0_use)

  # Target second moment (p x p)
  M0 <- crossprod(X0_use) / N0

  # helpers
  get_X_aug <- function(X) if (use_intercept) cbind(1, X) else X
  inv_M <- function(M) {
    # regularized inverse of a second-moment matrix
    solve(M + ridge * diag(nrow(M)))
  }

  # Pull what we need once
  beta_list <- lapply(fit$init_est, `[[`, "beta_init")  # each is length p

  for (k1 in 1:L) for (l1 in k1:L) {
    for (k2 in 1:L) for (l2 in k2:L) {
      ind1 <- .index_map(L, l1, k1)
      ind2 <- .index_map(L, l2, k2)

      # Augmented designs in p-dim space
      X_l1_aug <- get_X_aug(fit$X_list[[l1]])  # n_l1 x p
      X_k1_aug <- get_X_aug(fit$X_list[[k1]])  # n_k1 x p
      n_l1 <- nrow(X_l1_aug); n_k1 <- nrow(X_k1_aug)

      # Per-source second moments (p x p)
      M_l1 <- crossprod(X_l1_aug) / n_l1
      M_k1 <- crossprod(X_k1_aug) / n_k1

      # Inverses (regularized)
      iM_l1 <- inv_M(M_l1)
      iM_k1 <- inv_M(M_k1)

      # Coefs (length p), residual variances (scalars) from fit
      beta_l1 <- beta_list[[l1]]
      beta_k1 <- beta_list[[k1]]
      beta_l2 <- beta_list[[l2]]
      beta_k2 <- beta_list[[k2]]

      dev_l1 <- fit$dev_vec[l1]
      dev_k1 <- fit$dev_vec[k1]

      # Projections (p-vectors)
      Proj1 <- iM_l1 %*% M0 %*% beta_k1
      Proj3 <- iM_k1 %*% M0 %*% beta_l1

      # Conditionally needed inverses for l2/k2
      Proj2_l1 <- if (l2 == l1) {
        X_l2_aug <- get_X_aug(fit$X_list[[l2]])
        iM_l2    <- inv_M(crossprod(X_l2_aug) / nrow(X_l2_aug))
        iM_l2 %*% M0 %*% beta_k2
      } else rep(0, p)

      Proj2_k1 <- if (k2 == l1) {
        X_k2_aug <- get_X_aug(fit$X_list[[k2]])
        iM_k2    <- inv_M(crossprod(X_k2_aug) / nrow(X_k2_aug))
        iM_k2 %*% M0 %*% beta_l2
      } else rep(0, p)

      Proj4_l1 <- if (l2 == k1) {
        X_l2_aug <- get_X_aug(fit$X_list[[l2]])
        iM_l2    <- inv_M(crossprod(X_l2_aug) / nrow(X_l2_aug))
        iM_l2 %*% M0 %*% beta_k2
      } else rep(0, p)

      Proj4_k1 <- if (k2 == k1) {
        X_k2_aug <- get_X_aug(fit$X_list[[k2]])
        iM_k2    <- inv_M(crossprod(X_k2_aug) / nrow(X_k2_aug))
        iM_k2 %*% M0 %*% beta_l2
      } else rep(0, p)

      # Two dev-weighted quadratic pieces (use second moments in p-dim)
      val1 <- (dev_l1 / n_l1) * as.numeric(t(Proj1) %*% M_l1 %*% (Proj2_l1 + Proj2_k1))
      val2 <- (dev_k1 / n_k1) * as.numeric(t(Proj3) %*% M_k1 %*% (Proj4_l1 + Proj4_k1))

      # Covariance term on the target sample (preds use X0_use and p-dim betas)
      P1 <- as.numeric(X0_use %*% beta_l1) * as.numeric(X0_use %*% beta_k1)
      P2 <- as.numeric(X0_use %*% beta_l2) * as.numeric(X0_use %*% beta_k2)
      val3 <- mean((P1 - mean(P1)) * (P2 - mean(P2))) / N0

      Var_Gamma[ind1, ind2] <- val1 + val2 + val3
    }
  }

  # Symmetrize just in case of small numeric asymmetry
  Var_Gamma <- (Var_Gamma + t(Var_Gamma)) / 2

  # Diagonal regularization
  diag_correction <- pmax(tau * diag(Var_Gamma), 1.0 / min(ns))
  Var_Gamma <- Var_Gamma + diag(diag_correction, nrow(Var_Gamma))

  Var_Gamma
}

.compute_Var_Gamma_hd <- function(fit, tau = 0.2) {
  L <- fit$L
  d <- fit$d
  gen_dim <- L * (L + 1) / 2
  Var_Gamma <- matrix(NA_real_, gen_dim, gen_dim)

  N <- nrow(fit$X0_use)
  Sigma0 <- crossprod(fit$X0_use) / N

  .Sigma_l <- function(l) {
    # If you want exact per-source covariance, store X_list in fit and use:
    # Xl <- if (fit$intercept) cbind(1, fit$X_list[[l]]) else fit$X_list[[l]]
    # crossprod(Xl) / nrow(Xl)
    Sigma0
  }


  # High-dimensional path uses SIHR projections and devs
  for (k1 in 1:L) for (l1 in k1:L) {
    for (k2 in 1:L) for (l2 in k2:L) {
      ind1 <- .index_map(L, l1, k1)
      ind2 <- .index_map(L, l2, k2)

      dev_l1 <- fit$init_est[[l1]]$dev
      dev_k1 <- fit$init_est[[k1]]$dev

      Proj1    <- fit$Proj_array[l1, k1, ]
      Proj2_l1 <- if (l2 == l1) fit$Proj_array[l2, k2, ] else rep(0, d)
      Proj2_k1 <- if (k2 == l1) fit$Proj_array[k2, l2, ] else rep(0, d)
      val1 <- (dev_l1 / nrow(Sigma0)) * as.numeric(t(Proj1) %*% .Sigma_l(l1) %*% (Proj2_l1 + Proj2_k1))

      Proj3    <- fit$Proj_array[k1, l1, ]
      Proj4_l1 <- if (l2 == k1) fit$Proj_array[l2, k2, ] else rep(0, d)
      Proj4_k1 <- if (k2 == k1) fit$Proj_array[k2, l2, ] else rep(0, d)
      val2 <- (dev_k1 / nrow(Sigma0)) * as.numeric(t(Proj3) %*% .Sigma_l(k1) %*% (Proj4_l1 + Proj4_k1))

      P1 <- as.numeric(fit$X0_use %*% Proj1) * as.numeric(fit$X0_use %*% Proj3)
      P2 <- as.numeric(fit$X0_use %*% Proj2_l1) * as.numeric(fit$X0_use %*% Proj4_k1)
      val3 <- mean((P1 - mean(P1)) * (P2 - mean(P2))) / nrow(fit$X0_use)

      Var_Gamma[ind1, ind2] <- val1 + val2 + val3
    }
  }


  diag_correction <- pmax(tau * diag(Var_Gamma), 1.0 / nrow(fit$X0_use))
  Var_Gamma + diag(diag_correction)
}


######################################################################
################################ DRoL ################################
######################################################################
#### do we need CV here for xgb both learner?

# --------------------------- Bias-Correction Term ---------------------------
.bias_correct_term <- function(fk, fl, wl, Xl, yl) {
  wl <- wl$predict(Xl)
  fk <- fk$predict(Xl)
  fl <- fl$predict(Xl)
  mean(wl * fk * (fl - yl))
}
