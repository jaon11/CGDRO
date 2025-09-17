###### main function to run the app ######
cgdro <- function(X_list, y_list, X0 = NULL, loading_mat = NULL,
                  family = c("drlm_reg", "drlm_cls", "drol"),
                  f_learner = c("linear", "xgb", "xgb.cv", "high_d"),
                  w_learner = c("linear", "xgb", "xgb.cv","kliep"), ...){
  family <- match.arg(family)
  f_learner <- match.arg(f_learner)
  w_learner <- match.arg(w_learner)

  if (family == "drlm_reg") {
    if (is.null(loading_mat)) stop("For drlm_reg, loading_mat must be provided.")
    fit <- fit_drlm_reg(X_list, y_list, loading_mat, X0, ...)
  } else if (family == "drlm_cls") {
    fit <- fit_drlm_cls(X_list, y_list, X0,
                        f_learner, w_learner, ...)
  } else if (family == "drol") {
    fit <- fit_drol(X_list, y_list, X0,
                    f_learner, w_learner, ...)
  }
  fit
}

predict <- function(fit){
  if (!is.list(fit) || is.null(fit$family)) stop("fit must be a list returned by cgdro()")
  if (fit$family == "drlm_reg") {
    predict_drlm_reg(fit)
  } else if (fit$family == "drlm_cls") {
    predict_drlm_cls(fit)
  } else if (fit$family == "drol") {
    predict_drol(fit)
  }
}

infer <- function(fit, M = 50, alpha = 0.05, ...){
  if (!is.list(fit) || is.null(fit$family)) stop("fit must be a list returned by cgdro()")
  if (fit$family == "drlm_reg") {
    infer_drlm_reg(fit, M, alpha, ...)
  } else if (fit$family == "drlm_cls") {
    infer_drlm_cls(fit, M, alpha, ...)
  }
}

summary <- function(fit, infer = NULL, ...) {
  if (!is.list(fit) || is.null(fit$family)) stop("fit must be a list returned by cgdro()")
  if (fit$family == "drlm_reg") {
    summary_drlm_reg(fit, infer, ...)
  } else if (fit$family == "drlm_cls") {
    summary_drlm_cls(fit, infer, ...)
  } else if (fit$family == "drol") {
    print("Use predict() for DRoL Instead.")
  }
}
