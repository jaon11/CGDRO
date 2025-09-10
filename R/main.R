###### main function to run the app ######
cgdro <- function(X_list, y_list, X0 = NULL,
                  family = c("drlm_reg", "drlm_cls", "drol"),
                  f_learner = c("linear", "xgb", "xgb.cv", "high_d"),
                  w_learner = c("linear", "xgb", "xgb.cv"),
                  loading_mat = NULL, intercept = FALSE, intercept_loading = FALSE,
                  split = TRUE, max_iter = 1000, tol = 1e-6, check_dual = FALSE,
                  delta = 0, lambda = NULL, verbose = FALSE, seed = 123){
  family <- match.arg(family)
  f_learner <- match.arg(f_learner)
  w_learner <- match.arg(w_learner)

  if (family == "drlm_reg") {
    if (is.null(loading_mat)) stop("For drlm_reg, loading_mat must be provided.")
    fit <- fit_drlm_reg(X_list, y_list, loading_mat, X0, intercept,
                        intercept_loading, delta, lambda, verbose)
  } else if (family == "drlm_cls") {
    fit <- fit_drlm_cls(X_list, y_list, X0,
                        f_learner, w_learner,
                        split, max_iter, tol, check_dual,
                        verbose, seed)
  } else if (family == "drol") {
    fit <- fit_drol(X_list, y_list, X0,
                    f_learner, w_learner, seed)
  }
  fit
}

predict <- function(fit, bias_correct = TRUE, priors = NULL,
                    ridge = 1e-8, solver = c("ECOS", "SCS")){
  if (!is.list(fit) || is.null(fit$family)) stop("fit must be a list returned by cgdro()")
  if (fit$family == "drlm_reg") {
    predict_drlm_reg(fit)
  } else if (fit$family == "drlm_cls") {
    predict_proba_drlm_cls(fit)
  } else if (fit$family == "drol") {
    predict_drol(fit, bias_correct, priors, ridge, solver)
  }
}

infer <- function(fit, index = 1, M = 50, alpha = 0.05, parallel = FALSE, n_workers = 2, diag = TRUE,
                  tau = 0.2, alpha_thres = 0.01, threshold = 0) {
  if (!is.list(fit) || is.null(fit$family)) stop("fit must be a list returned by cgdro()")
  if (fit$family == "drlm_reg") {
    infer_drlm_reg(fit, M, alpha, tau, alpha_thres, threshold)
  } else if (fit$family == "drlm_cls") {
    infer_drlm_cls(fit, index, M, alpha, parallel, n_workers, diag)
  }
}

