###### main function to run the app ######
cgdro <- function(
    X_list, y_list,
    X0 = NULL,
    index = NULL,
    family   = c("reg_ld", "reg_hd", "reg_ml", "cls"),
    f_learner = c("linear", "xgb", "xgb.cv", "high_d"),
    w_learner = c("linear", "xgb", "xgb.cv", "ulsif"),
    loss_type = NULL,   # only used for reg_ld and reg_ml
    ...
) {
  family <- match.arg(family)

  # Handle loss_type only for reg_ld and reg_ml
  if (family %in% c("reg_ld", "reg_ml")) {
    if (is.null(loss_type)) loss_type <- "reward"
    loss_type <- match.arg(loss_type, c("reward", "squaredloss", "regret"))
  } else {
    if (!is.null(loss_type)) {
      warning("`loss_type` is ignored when family is not 'reg_ld' or 'reg_ml'.")
    }
  }

  # Match learners only for the families that use them
  if (family %in% c("reg_ml", "cls")) {
    f_learner <- match.arg(f_learner)
    w_learner <- match.arg(w_learner)
  }

  if (family == "reg_ld") {
    fit <- fit_reg_ld(
      X_list, y_list, X0,
      loss_type = loss_type, ...
    )

  } else if (family == "reg_hd") {
    if (is.null(index)) stop("`index` must be provided for high-dimensional regression (family = 'reg_hd').")
    fit <- fit_reg_hd(
      X_list, y_list, index, X0,
      ...  # no loss_type here
    )

  } else if (family == "reg_ml") {
    fit <- fit_reg_ml(
      X_list, y_list, X0,
      loss_type = loss_type,
      f_learner = f_learner,
      w_learner = w_learner,
      ...
    )

  } else if (family == "cls") {
    fit <- fit_cls(
      X_list, y_list, X0,
      f_learner = f_learner,
      w_learner = w_learner,
      ...
    )
  }

  fit
}


predict <- function(fit, X=NULL){
  if (!is.list(fit) || is.null(fit$family)) stop("fit must be a list returned by cgdro()")
  if (fit$family == "reg_ld") {
    predict_reg_ld(fit, X)
  } else if (fit$family == "reg_hd") {
    predict_reg_hd(fit, X)
  } else if (fit$family == "reg_ml") {
    predict_reg_ml(fit, X)
  } else if (fit$family == "cls") {
    predict_cls(fit, X)
  }
}

infer <- function(fit, M = 50, alpha = 0.05, ...){
  if (!is.list(fit) || is.null(fit$family)) stop("fit must be a list returned by cgdro()")
  if (fit$family == "reg_ld") {
    infer_reg_ld(fit, M, alpha, ...)
  } else if (fit$family == "reg_hd") {
    infer_reg_hd(fit, M, alpha, ...)
  } else if (fit$family == "cls") {
    infer_cls(fit, M, alpha, ...)
  }
}

summary <- function(fit, infer = NULL, ...) {
  if (!is.list(fit) || is.null(fit$family)) stop("fit must be a list returned by cgdro()")
  if (fit$family == "reg_ld") {
    summary_reg_ld(fit, infer, ...)
  } else if (fit$family == "reg_hd") {
    summary_reg_hd(fit, infer, ...)
  } else if (fit$family == "cls") {
    summary_cls(fit, infer, ...)
  } else if (fit$family == "reg_ml") {
    print("Use predict() for reg_ml Instead.")
  }
}
