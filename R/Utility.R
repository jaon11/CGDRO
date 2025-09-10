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


# =====================================================================
# Helpers
# =====================================================================







# ---- Probability learner (multi-class) using nnet::multinom ----
.compute_proba_once <- function(X, y, X0, learner = "xgb", split = FALSE,
                                proba_params = NULL, seed = 123) {
  set.seed(seed)
  y <- as.factor(y)  # required by multinom
  num_class <- length(levels(y))

  # train/predict wrappers
  if (learner == "xgb.cv") {
    train_model <- function(X, y) {
      if (!requireNamespace("xgboost", quietly = TRUE)) stop("Need xgboost")

      # --- Preprocess ----------------------------------------------------------
      X_mat <- if (is.data.frame(X)) {
        mm <- stats::model.matrix(~ . - 1, data = X); storage.mode(mm) <- "double"; mm
      } else data.matrix(X)

      y_fac <- as.factor(y)
      classes <- levels(y_fac)
      y_int <- as.integer(y_fac) - 1L

      # optional weight & seed (mirrors your style of reading from parent env)
      sw   <- get0("sample_weight", ifnotfound = NULL, inherits = TRUE)
      seed <- get0("seed",           ifnotfound = 1L,   inherits = TRUE)

      dtrain <- xgboost::xgb.DMatrix(data = X_mat, label = y_int, weight = sw, missing = NA)

      base_params <- list(
        objective   = "multi:softprob",
        eval_metric = "mlogloss",
        num_class   = length(classes)
      )

      # --- Branch 1: use provided proba_params --------------------------------
      pp <- get0("proba_params", ifnotfound = NULL, inherits = TRUE)

      if (!is.null(pp)) {
        if (!is.list(pp)) stop("`proba_params` must be a named list if provided.")
        # pull nrounds out
        nrounds <- if (!is.null(pp$nrounds)) as.integer(pp$nrounds) else 200L
        pp$nrounds <- NULL

        params <- utils::modifyList(base_params, pp)

        m <- xgboost::xgb.train(
          params  = params,
          data    = dtrain,
          nrounds = nrounds,
          verbose = 0
        )

        return(list(model = m, classes = classes))
      }

      # --- Branch 2: 5-fold stratified CV to choose params --------------------
      set.seed(seed)

      # a sensible default grid (adjust as you like)
      param_grid <- expand.grid(
        eta               = c(0.05, 0.1, 0.2),
        max_depth         = c(4, 6, 8),
        min_child_weight  = c(1, 3),
        subsample         = c(0.8, 1.0),
        colsample_bytree  = c(0.8, 1.0),
        lambda            = c(1, 5),
        alpha             = c(0, 1),
        KEEP.OUT.ATTRS    = FALSE,
        stringsAsFactors  = FALSE
      )

      best_score      <- Inf
      best_nrounds    <- 200L
      best_params_row <- NULL

      # We’ll let early stopping pick nrounds; run up to max_nrounds
      max_nrounds <- 1000L
      early_stop  <- 50L

      for (i in seq_len(nrow(param_grid))) {
        row <- as.list(param_grid[i, , drop = FALSE])
        params_try <- utils::modifyList(base_params, row)

        cv <- xgboost::xgb.cv(
          params                = params_try,
          data                  = dtrain,
          nrounds               = max_nrounds,
          nfold                 = 5,
          stratified            = TRUE,
          early_stopping_rounds = early_stop,
          verbose               = 0,
          maximize              = FALSE,   # lower mlogloss is better
          showsd                = TRUE,
          seed                  = seed
        )

        # xgb.cv stores the best logloss in evaluation_log; use cv$best_iteration
        cv_best_iter  <- cv$best_iteration
        cv_best_score <- cv$evaluation_log$test_mlogloss_mean[cv_best_iter]

        if (!is.na(cv_best_score) && cv_best_score < best_score) {
          best_score      <- cv_best_score
          best_nrounds    <- cv_best_iter
          best_params_row <- row
        }
      }

      if (is.null(best_params_row)) {
        # fallback (shouldn’t happen unless data is degenerate)
        best_params_row <- as.list(param_grid[1, , drop = FALSE])
        best_nrounds    <- 200L
      }

      final_params <- utils::modifyList(base_params, best_params_row)

      m <- xgboost::xgb.train(
        params  = final_params,
        data    = dtrain,
        nrounds = best_nrounds,
        verbose = 0
      )

      list(model = m, classes = classes)
    }

    predict_prob <- function(m, X) {
      X_mat <- if (is.data.frame(X)) {
        mm <- stats::model.matrix(~ . - 1, data = X); storage.mode(mm) <- "double"; mm
      } else data.matrix(X)

      pred <- predict(m$model, xgboost::xgb.DMatrix(X_mat, missing = NA))
      K <- length(m$classes)
      probs <- matrix(pred, ncol = K, byrow = TRUE)
      colnames(probs) <- m$classes
      probs
    }
  }else if (learner == "xgb"){
    # ---------- Multiclass XGBoost (no CV), same style as your xib ----------
    # train_model(): pull nrounds out of params, modifyList with base_params, train
    # predict_prob(): return n x K probability matrix (cols named by class levels)

    train_model <- function(X, y, params = NULL) {
      if (!requireNamespace("xgboost", quietly = TRUE)) stop("Need xgboost")

      # 1) nrounds default + strip from params
      nrounds <- if (!is.null(params) && !is.null(params$nrounds)) params$nrounds else 200L
      params_no_nrounds <- params
      if (!is.null(params_no_nrounds)) params_no_nrounds$nrounds <- NULL
      if (!is.null(params_no_nrounds) && !is.list(params_no_nrounds)) {
        stop("`params` must be a named list when provided.")
      }

      # 2) targets -> integers 0..K-1 + class levels
      y_fac <- as.factor(y)
      classes <- levels(y_fac)
      y_int <- as.integer(y_fac) - 1L

      # 3) base params + user overrides
      base_params <- list(
        objective   = "multi:softprob",
        eval_metric = "mlogloss",
        num_class   = length(classes)
      )
      if (!is.null(params_no_nrounds)) {
        base_params <- utils::modifyList(base_params, params_no_nrounds)
      }

      # 4) data -> DMatrix (optionally with sample weights)
      X_mat <- if (is.data.frame(X)) {
        mm <- stats::model.matrix(~ . - 1, data = X); storage.mode(mm) <- "double"; mm
      } else {
        data.matrix(X)
      }
      sw <- get0("sample_weight", ifnotfound = NULL, inherits = TRUE)
      dtrain <- xgboost::xgb.DMatrix(data = X_mat, label = y_int, weight = sw, missing = NA)

      # 5) train
      m <- xgboost::xgb.train(
        params  = base_params,
        data    = dtrain,
        nrounds = nrounds,
        verbose = 0
      )

      list(
        model   = m,
        params  = base_params,
        nrounds = nrounds,
        classes = classes
      )
    }

    predict_prob <- function(m, X) {
      if (is.null(m$model) || is.null(m$classes)) {
        stop("`m` must be the object returned by train_model() with $model and $classes.")
      }
      X_mat <- if (is.data.frame(X)) {
        mm <- stats::model.matrix(~ . - 1, data = X); storage.mode(mm) <- "double"; mm
      } else {
        data.matrix(X)
      }
      raw <- predict(m$model, xgboost::xgb.DMatrix(X_mat, missing = NA))
      K <- length(m$classes)
      probs <- matrix(raw, ncol = K, byrow = TRUE)
      colnames(probs) <- m$classes
      probs
    }

  }else if (learner == "linear"){
    train_model <- function(X, y) {
      df <- data.frame(y = y, X)
      # you can pass decay / maxit in proba_params if desired
      args <- c(list(formula = y ~ ., data = df, trace = FALSE, MaxNWts = 100000),
                proba_params)
      do.call(multinom, args)
    }
    predict_prob <- function(m, X) {
      predict(m, newdata = data.frame(X), type = "probs")
    }

  }else {
    stop("Unsupported prob_learner: use learner = 'nnet' for multi-class.")
  }

  if (split) {
    n <- nrow(X)
    indA <- sample(n, floor(n/2))
    indB <- setdiff(seq_len(n), indA)

    mA <- train_model(X[indA, ], y[indA])
    mB <- train_model(X[indB, ], y[indB])

    predAB <- predict_prob(mA, X[indB, ])  # on B using A
    predBA <- predict_prob(mB, X[indA, ])  # on A using B
    predX0 <- (predict_prob(mA, X0) + predict_prob(mB, X0)) / 2

    predX <- matrix(0, nrow = n, ncol = num_class)
    predX[indA, ] <- predBA
    predX[indB, ] <- predAB
  } else {
    m <- train_model(X, y)
    predX  <- predict_prob(m, X)
    predX0 <- predict_prob(m, X0)
  }

  # ensure matrix (n, C)
  predX  <- as.matrix(predX)
  predX0 <- as.matrix(predX0)

  # Clip probs to avoid extremes
  predX  <- pmin(pmax(predX, 1e-6), 1 - 1e-6)
  predX0 <- pmin(pmax(predX0, 1e-6), 1 - 1e-6)

  list(predX = predX, predX0 = predX0)
}

# ---- Density-ratio learner (binary) via glmnet (binomial) ----
.compute_density_once <- function(X, X0, learner = "linear", split = FALSE,
                                  density_params = NULL, seed = 123) {
  set.seed(seed)
  if (learner == "linear") {
    train_model <- function(X, y) {
      # y should be numeric 0/1
      cv.glmnet(X, y, family = "binomial", type.measure = "deviance")
    }
    predict_prob <- function(m, X) as.numeric(predict(m, X, s = "lambda.min", type = "response"))
  } else if (learner == "xgb") {
    # ---------- Multiclass XGBoost (no CV), same style as your xib ----------
    # train_model(): pull nrounds out of params, modifyList with base_params, train
    # predict_prob(): return n x K probability matrix (cols named by class levels)

    train_model <- function(X, y, params = NULL) {
      if (!requireNamespace("xgboost", quietly = TRUE)) stop("Need xgboost")
      X_mat <- if (is.data.frame(X)) {
        mm <- stats::model.matrix(~ . - 1, data = X); storage.mode(mm) <- "double"; mm
      } else data.matrix(X)

      # Binary labels -> {0,1}
      if (is.factor(y) || is.character(y) || is.logical(y)) y <- as.integer(as.factor(y)) - 1L
      y <- as.numeric(y)
      if (!all(y %in% c(0, 1))) stop("y must be binary (0/1, logical, or two-level factor).")

      # Pull nrounds out of params (default 200), and don't keep it in the params list
      nrounds <- if (!is.null(params) && !is.null(params$nrounds)) params$nrounds else 200
      # Base params without nrounds
      base_params <- list(objective = "binary:logistic", eval_metric = "logloss")
      if (!is.null(params)) {
        params_no_nrounds <- params
        params_no_nrounds$nrounds <- NULL
        base_params <- modifyList(base_params, params_no_nrounds)
      }

      dtrain <- xgboost::xgb.DMatrix(data = as.matrix(X_mat), label = y)
      m <- xgboost::xgb.train(params = base_params, data = dtrain,
                              nrounds = nrounds, verbose = 0)

      list(
        model   = m,
        params  = base_params,
        nrounds = nrounds
      )
    }

    predict_prob <- function(m, X) {
      if (is.null(m$model)) {
        stop("`m` must be the object returned by train_model() with $model.")
      }
      X_mat <- if (is.data.frame(X)) {
        mm <- stats::model.matrix(~ . - 1, data = X); storage.mode(mm) <- "double"; mm
      } else {
        data.matrix(X)
      }
      probs <- predict(m$model, xgboost::xgb.DMatrix(X_mat, missing = NA))
      probs
    }


  }else if (learner == "xgb.cv") {
    train_model <- function(X, y, params = NULL) {
      if (!requireNamespace("xgboost", quietly = TRUE)) stop("Need xgboost")

      # --- Preprocess ----------------------------------------------------------
      # Encode X like glm (handles factors/characters)
      X_mat <- if (is.data.frame(X)) {
        mm <- stats::model.matrix(~ . - 1, data = X); storage.mode(mm) <- "double"; mm
      } else data.matrix(X)

      # Binary labels -> {0,1}
      if (is.factor(y) || is.character(y) || is.logical(y)) y <- as.integer(as.factor(y)) - 1L
      y <- as.numeric(y)
      if (!all(y %in% c(0, 1))) stop("y must be binary (0/1, logical, or two-level factor).")

      # Optional weight & seed
      sw   <- get0("sample_weight", ifnotfound = NULL, inherits = TRUE)
      seed <- get0("seed",           ifnotfound = 1L,   inherits = TRUE)

      dtrain <- xgboost::xgb.DMatrix(data = X_mat, label = y, weight = sw, missing = NA)

      # --- Base params + user overrides (pull nrounds out) ---------------------
      nrounds <- if (!is.null(params) && !is.null(params$nrounds)) params$nrounds else 200L
      params_no_nrounds <- params
      if (!is.null(params_no_nrounds)) {
        if (!is.list(params_no_nrounds)) stop("`params` must be a named list when provided.")
        params_no_nrounds$nrounds <- NULL
      }

      base_params <- list(objective = "binary:logistic", eval_metric = "logloss")
      if (!is.null(params_no_nrounds)) {
        base_params <- utils::modifyList(base_params, params_no_nrounds)
      }

      # --- Branch 1: direct training if params provided ------------------------
      if (!is.null(params)) {
        m <- xgboost::xgb.train(
          params  = base_params,
          data    = dtrain,
          nrounds = nrounds,
          verbose = 0
        )
        return(list(model = m, params = base_params, nrounds = nrounds))
      }

      # --- Branch 2: 5-fold stratified CV (when params == NULL) ----------------
      set.seed(seed)

      # A compact but useful grid (tune as needed)
      param_grid <- expand.grid(
        eta               = c(0.05, 0.1, 0.2),
        max_depth         = c(3, 5, 7),
        min_child_weight  = c(1, 3),
        subsample         = c(0.8, 1.0),
        colsample_bytree  = c(0.8, 1.0),
        lambda            = c(1, 5),
        alpha             = c(0, 1),
        KEEP.OUT.ATTRS    = FALSE,
        stringsAsFactors  = FALSE
      )

      best_score      <- Inf
      best_nrounds    <- 200L
      best_params_row <- NULL

      max_nrounds <- 1000L
      early_stop  <- 50L

      for (i in seq_len(nrow(param_grid))) {
        row <- as.list(param_grid[i, , drop = FALSE])
        params_try <- utils::modifyList(base_params, row)

        cv <- xgboost::xgb.cv(
          params                = params_try,
          data                  = dtrain,
          nrounds               = max_nrounds,
          nfold                 = 5,
          stratified            = TRUE,
          early_stopping_rounds = early_stop,
          verbose               = 0,
          maximize              = FALSE,  # minimize logloss
          showsd                = TRUE,
          seed                  = seed
        )

        cv_best_iter <- cv$best_iteration
        if (is.null(cv_best_iter) || is.na(cv_best_iter) || cv_best_iter <= 0) {
          cv_best_iter <- nrow(cv$evaluation_log)
        }
        cv_best_score <- cv$evaluation_log$test_logloss_mean[cv_best_iter]

        if (!is.na(cv_best_score) && cv_best_score < best_score) {
          best_score      <- cv_best_score
          best_nrounds    <- cv_best_iter
          best_params_row <- row
        }
      }

      if (is.null(best_params_row)) {
        best_params_row <- as.list(param_grid[1, , drop = FALSE])
        best_nrounds    <- 200L
      }

      final_params <- utils::modifyList(base_params, best_params_row)

      m <- xgboost::xgb.train(
        params  = final_params,
        data    = dtrain,
        nrounds = best_nrounds,
        verbose = 0
      )

      list(model = m, params = final_params, nrounds = best_nrounds)
    }

    predict_prob <- function(m, X) {
      if (is.null(m$model)) stop("`m` must be the object returned by train_model() with $model.")
      X_mat <- if (is.data.frame(X)) {
        mm <- stats::model.matrix(~ . - 1, data = X); storage.mode(mm) <- "double"; mm
      } else data.matrix(X)
      as.numeric(predict(m$model, xgboost::xgb.DMatrix(X_mat, missing = NA)))
    }


  }else stop("Unknown density_learner: ", learner)

  if (split) {
    n <- nrow(X)
    indA <- sample(n, floor(n/2))
    indB <- setdiff(seq_len(n), indA)

    XA <- rbind(X[indA, ], X0); yA <- c(rep(0, length(indA)), rep(1, nrow(X0)))
    XB <- rbind(X[indB, ], X0); yB <- c(rep(0, length(indB)), rep(1, nrow(X0)))

    mA <- train_model(XA, yA); mB <- train_model(XB, yB)

    pAB <- predict_prob(mA, X[indB, ])
    pBA <- predict_prob(mB, X[indA, ])

    omegaB <- (pAB/(1 - pAB)) * (length(indA)/nrow(X0))
    omegaA <- (pBA/(1 - pBA)) * (length(indB)/nrow(X0))
    omegaX <- numeric(n); omegaX[indA] <- omegaA; omegaX[indB] <- omegaB
  } else {
    Xc <- rbind(X, X0); yc <- c(rep(0, nrow(X)), rep(1, nrow(X0)))
    m <- train_model(Xc, yc)
    p <- predict_prob(m, X)
    omegaX <- (p/(1 - p)) * (nrow(X)/nrow(X0))
  }

  pmin(pmax(omegaX, 1e-3), 1e3)
}

# ---- Fit probas/densities for all sources ----
.fit_proba_density <- function(X_list, y_list, X0, prob_learner, density_learner,
                               split, proba_params_list, density_params_list, seed=123) {
  L <- length(X_list)
  probaX_list  <- vector("list", L)
  probaX0_list <- vector("list", L)
  omegaX_list  <- vector("list", L)
  for (l in seq_len(L)) {
    pr <- .compute_proba_once(X_list[[l]], y_list[[l]], X0,
                              learner = prob_learner, split = split,
                              proba_params = proba_params_list[[l]], seed = seed)
    probaX_list[[l]]  <- pr$predX         # (n_l, C)
    probaX0_list[[l]] <- pr$predX0        # (n0, C)
  }
  for (l in seq_len(L)) {
    omegaX_list[[l]] <- .compute_density_once(X_list[[l]], X0,
                                              learner = density_learner, split = split,
                                              density_params = density_params_list[[l]], seed = seed)
  }
  list(probaX_list = probaX_list, probaX0_list = probaX0_list, omegaX_list = omegaX_list)
}



.train_lasso <- function(X, y, intercept = FALSE, lambda_val = NULL, max_iter = 1e5) {
  X <- as.matrix(X); y <- as.numeric(y)
  X_aug <- if (intercept) cbind(1, X) else X

  if (is.null(lambda_val) || identical(lambda_val, "CV.min")) {
    cvfit <- cv.glmnet(X_aug, y, family = "gaussian",
                       alpha = 1, intercept = FALSE, standardize = FALSE)
    as.numeric(predict(cvfit, type = "coefficients", s = "lambda.min"))[-1]
  } else if (identical(lambda_val, "CV")) {
    cvfit <- cv.glmnet(X_aug, y, family = "gaussian",
                       alpha = 1, intercept = FALSE, standardize = FALSE)
    lam <- cvfit$lambda.1se
    as.numeric(predict(cvfit, type = "coefficients", s = lam))[-1]
  } else {
    fit <- glmnet(X_aug, y, family = "gaussian", alpha = 1,
                  lambda = lambda_val, intercept = FALSE,
                  standardize = FALSE, maxit = max_iter)
    as.numeric(coef(fit))[-1]
  }
}



# --------------------------- Outcome Learners ---------------------------

.fit_outcome <- function(X, y, learner = "xgb", params = NULL, sample_weight = NULL) {
  if (is.null(sample_weight)) sample_weight <- rep(1, length(y))
  X <- as.matrix(X); y <- as.numeric(y)

  if (learner == "linear") {
    df <- data.frame(y = y, X)
    m <- lm(y ~ ., data = df, weights = sample_weight)
    pred <- function(X) as.numeric(predict(m, newdata = data.frame(X)))
    return(list(predict = pred, model = m))
  }

  if (learner == "xgb") {
    if (!requireNamespace("xgboost", quietly = TRUE)) stop("Need xgboost")
    base_params <- list(
      objective = "reg:squarederror",
      eval_metric = "rmse"
    )
    if (!is.null(params)) base_params <- modifyList(base_params, params)

    dtrain <- xgboost::xgb.DMatrix(data = X, label = y, weight = sample_weight)
    nrounds <- if (!is.null(params$nrounds)) params$nrounds else 200

    m <- xgboost::xgb.train(params = base_params, data = dtrain,
                            nrounds = nrounds, verbose = 0)
    pred <- function(X) as.numeric(predict(m, xgboost::xgb.DMatrix(as.matrix(X))))
    return(list(predict = pred, model = m))
  }

  stop("Unknown outcome learner: ", learner)
}

# --------------------------- Density Learners ---------------------------

.fit_density <- function(X, X_target, learner = "xgb", params = NULL) {
  X <- as.matrix(X); X_target <- as.matrix(X_target)
  ratio <- nrow(X) / nrow(X_target)

  Xc <- rbind(X, X_target)
  yc <- c(rep(0, nrow(X)), rep(1, nrow(X_target)))

  if (learner == "linear") {
    df <- data.frame(y = yc, Xc)
    m <- glm(y ~ ., data = df, family = binomial())
    pred <- function(X) {
      p1 <- as.numeric(predict(m, newdata = data.frame(X), type = "response"))
      p1 <- pmin(pmax(p1, 1e-8), 1 - 1e-8)
      (p1 / (1 - p1)) * ratio
    }
    return(list(predict = pred, model = m, sample_ratio = ratio))
  }

  if (learner == "xgb") {
    if (!requireNamespace("xgboost", quietly = TRUE)) {
      stop("Density learner 'xgb' requested but package 'xgboost' is not installed.")
    }
    # Pull nrounds out of params (default 200), and don't keep it in the params list
    nrounds <- if (!is.null(params) && !is.null(params$nrounds)) params$nrounds else 200
    # Base params without nrounds
    base_params <- list(objective = "binary:logistic", eval_metric = "logloss")
    if (!is.null(params)) {
      params_no_nrounds <- params
      params_no_nrounds$nrounds <- NULL
      base_params <- modifyList(base_params, params_no_nrounds)
    }

    dtrain <- xgboost::xgb.DMatrix(data = as.matrix(Xc), label = yc)
    m <- xgboost::xgb.train(params = base_params, data = dtrain,
                            nrounds = nrounds, verbose = 0)
    pred <- function(X) {
      p1 <- as.numeric(predict(m, xgboost::xgb.DMatrix(as.matrix(X))))
      p1 <- pmin(pmax(p1, 1e-8), 1 - 1e-8)
      (p1 / (1 - p1)) * ratio
    }
    return(list(predict = pred, model = m, sample_ratio = ratio))
  }

  stop("Unknown density learner: ", learner)
}


# ================================================================
# Unified learners: learn_f (outcome) and learn_w (density ratio)
# ================================================================

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
                           eta              = c(0.05, 0.1),
                           max_depth        = c(4, 6),
                           min_child_weight = c(1, 3),
                           subsample        = c(0.8, 1.0),
                           colsample_bytree = c(0.8, 1.0),
                           lambda           = c(1, 5),
                           alpha            = c(0, 1),
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

      model <- NULL          # coefficient vector aligned with (possibly) augmented X
      p_fit <- NULL          # ncol used at fit time (after augmentation)
      used_intercept <- FALSE

      fit <- function(X, y, intercept = FALSE) {
        Xm <- .mm_if_needed(X)
        if (intercept) Xm <- cbind(1, Xm)        # your rule
        y  <- as.numeric(y)

        if (is.null(lambda_val) || identical(lambda_val, "CV.min")) {
          cvfit <- glmnet::cv.glmnet(
            x = Xm, y = y, family = "gaussian",
            alpha = 1, intercept = FALSE, standardize = FALSE,
            weights = sample_weight
          )
          beta <- as.numeric(stats::predict(cvfit, type = "coefficients", s = "lambda.min"))[-1]

        } else if (identical(lambda_val, "CV")) {
          cvfit <- glmnet::cv.glmnet(
            x = Xm, y = y, family = "gaussian",
            alpha = 1, intercept = FALSE, standardize = FALSE,
            weights = sample_weight
          )
          lam  <- cvfit$lambda.1se
          beta <- as.numeric(stats::predict(cvfit, type = "coefficients", s = lam))[-1]

        } else {
          # numeric (scalar or vector) lambda
          fit0 <- glmnet::glmnet(
            x = Xm, y = y, family = "gaussian",
            alpha = 1, lambda = lambda_val,
            intercept = FALSE, standardize = FALSE,
            weights = sample_weight
          )
          # if lambda_val has length > 1, take the first by default
          s_use <- if (length(lambda_val) == 1) lambda_val else lambda_val[1]
          beta  <- as.numeric(stats::coef(fit0, s = s_use))[-1]
        }

        model <<- list(beta = beta, intercept = intercept)
        p_fit <<- ncol(Xm)
        used_intercept <<- intercept
        invisible(NULL)
      }

      predict <- function(Xnew) {
        if (is.null(model)) stop("Call $fit(X, y, intercept=...) before predicting.")
        Xn <- .mm_if_needed(Xnew)
        if (model$intercept) Xn <- cbind(1, Xn)
        if (ncol(Xn) != p_fit)
          stop(sprintf("Column mismatch: trained with p=%d, got p=%d.", p_fit, ncol(Xn)))
        as.numeric(Xn %*% model$beta)
      }

      return(list(
        fit = fit,
        predict = predict,
        model = function() model,
        coef = function() model$beta,
        mode = "reg",
        lambda_val = function() lambda_val,
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
.learn_w <- function(learner = c("linear", "xgb", "xgb.cv"),
                    params = NULL,
                    seed = 123) {
  learner <- match.arg(learner)
  set.seed(seed)

  model <- NULL
  ratio <- NA_real_
  params_used <- NULL
  nrounds_used <- NA_integer_

  fit <- function(X, X_target) {
    X  <- .mm_if_needed(X)
    X0 <- .mm_if_needed(X_target)
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

  predict <- function(Xnew) {
    if (is.null(model)) stop("Call $fit(X, X_target) before predicting ratios.")
    Xn <- .mm_if_needed(Xnew)
    p1 <- if (inherits(model, "cv.glmnet")) {
      as.numeric(stats::predict(model, Xn, s = "lambda.min", type = "response"))
    } else {
      as.numeric(stats::predict(model, newdata = Xn, missing = NA))  # generic -> xgb method
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

.opt_weight <- function(Gamma, delta = 0) {
  L <- ncol(Gamma)
  ed <- eigen((Gamma + t(Gamma)) / 2, symmetric = TRUE)
  lam <- pmax(ed$values, 1e-3)
  Gamma_pos <- ed$vectors %*% diag(lam, nrow = L) %*% t(ed$vectors)

  v <- CVXR::Variable(L)
  G <- Gamma_pos + delta * diag(L)
  prob <- CVXR::Problem(
    CVXR::Minimize(CVXR::quad_form(v, G)),
    list(CVXR::sum_entries(v) == 1, v >= 0)
  )
  res <- try(CVXR::solve(prob, solver = "ECOS"), silent = TRUE)
  if (inherits(res, "try-error") || res$status %in% c("infeasible", "unbounded")) {
    res <- CVXR::solve(prob, solver = "SCS")
  }
  w <- as.numeric(res$getValue(v))
  w[is.na(w)] <- 0
  w[w < 0] <- 0
  s <- sum(w); if (s <= 0) stop("Weight optimization failed.")
  w / s
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

.compute_Var_Gamma <- function(fit, tau = 0.2) {
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
