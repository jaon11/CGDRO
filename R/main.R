#' CGDRO Family: consolidated arguments per family and complete examples
#'
#' This page supplements the function-level help pages by listing the
#' \strong{family-specific arguments} for \code{\link{cgdro_}()}, \code{\link{infer_cgdro_}()},
#' and \code{\link{summary_cgdro_}()}, and by providing end-to-end \strong{examples} for each
#' supported family. Use it as the single place to discover which extra arguments are
#' recognized under different \code{family} settings and how to run complete workflows.
#'
#' @section Families:
#' \preformatted{
#' family <- c("reg_ld", "reg_hd", "reg_ml", "cls")
#' }
#'
#' \tabular{lll}{
#' \strong{Family} \tab \strong{Description} \tab \strong{Statistical Inference} \cr
#' \code{reg_ld} \tab Linear prediction model (low-dimensional) \tab yes \cr
#' \code{reg_hd} \tab High-dimensional linear model            \tab yes \cr
#' \code{reg_ml} \tab Machine learning prediction model        \tab no  \cr
#' \code{cls}    \tab Linear model for classification tasks    \tab yes \cr
#' }
#'
#'
#' @section Arguments by family:
#' \describe{
#'
#' \item{\strong{family = "reg_ld"}}{
#'   \itemize{
#'     \item \code{cgdro_()}:
#'       \describe{
#'         \item{\code{intercept}}{Whether to include intercept in outcome models. Default \code{FALSE}.}
#'         \item{\code{delta}}{Regularization parameter in weight optimization. Default \code{0}.}
#'       }
#'     \item \code{infer_cgdro_()}:
#'       \describe{
#'         \item{\code{tau}}{Variance inflation parameter. Default \code{0.2}.}
#'         \item{\code{alpha_thres}}{Threshold for small eigenvalues. Default \code{0.01}.}
#'         \item{\code{threshold}}{Threshold for eigenvalue truncation. Default \code{0}.}
#'       }
#'     \item \code{summary_cgdro_()}:
#'       \describe{
#'         \item{\code{index}}{Index of the coefficients of interest.}
#'       }
#'   }
#' }
#'
#' \item{\strong{family = "reg_hd"}}{
#'   \itemize{
#'     \item \code{cgdro_()}:
#'       \describe{
#'         \item{\code{intercept}}{Whether to include intercept in outcome models. Default \code{FALSE}.}
#'         \item{\code{intercept_loading}}{Whether to include intercept in loading matrix defined by \code{index}. Default \code{FALSE}.}
#'         \item{\code{delta}}{Regularization parameter in weight optimization. Default \code{0}.}
#'         \item{\code{lambda}}{Regularization parameter in high-dimensional outcome models.If \code{"CV.min"} or \code{"CV.1se"}, use cross-validation to select. Default \code{NULL}.}
#'       }
#'     \item \code{infer_cgdro_()}:
#'       \describe{
#'         \item{\code{tau}}{Variance inflation parameter. Default \code{0.2}.}
#'         \item{\code{alpha_thres}}{Threshold for small eigenvalues. Default \code{0.01}.}
#'         \item{\code{threshold}}{Threshold for eigenvalue truncation. Default \code{0}.}
#'       }
#'   }
#' }
#'
#' \item{\strong{family = "reg_ml"}}{
#'   \itemize{
#'     \item \code{cgdro_()}:
#'       \describe{
#'         \item{\code{bias_correct}}{Whether to use bias-corrected estimator for \eqn{\Gamma}. Default \code{TRUE}.}
#'         \item{\code{priors}}{Optional list with two elements: prior weight vector (length \eqn{L}) and radius (nonnegative scalar). Default \code{NULL} (no prior).}
#'         \item{\code{ridge}}{Ridge regularization parameter (nonnegative scalar) for numerical stability. Default \code{1e-8}.}
#'         \item{\code{solver}}{Solver for the quadratic program. Options: \code{"ECOS"}, \code{"SCS"}. Default \code{"ECOS"}.}
#'         \item{\code{seed}}{Random seed for reproducibility in data splitting. Default \code{123}.}
#'       }
#'   }
#' }
#'
#' \item{\strong{family = "cls"}}{
#'   \itemize{
#'     \item \code{cgdro_()}:
#'       \describe{
#'         \item{\code{split}}{Whether to use sample-splitting in outcome/density estimation. Default \code{TRUE}.}
#'         \item{\code{max_iter}}{Maximum number of iterations. Default \code{1000}.}
#'         \item{\code{tol}}{Tolerance for convergence. Default \code{1e-6}.}
#'         \item{\code{Check_dual}}{Whether to compute duality gap every 50 iterations. Default \code{FALSE}.}
#'         \item{\code{seed}}{Random seed for reproducibility in data splitting. Default \code{123}.}
#'       }
#'     \item \code{infer_cgdro_()}:
#'       \describe{
#'         \item{\code{parallel}}{Whether to use parallel computing. Default \code{FALSE}.}
#'         \item{\code{n_workers}}{Number of workers for parallel computing. Default \code{4}.}
#'         \item{\code{diag}}{Whether to use diagonal approximation for covariance estimation. Default \code{TRUE}.}
#'       }
#'     \item \code{summary_cgdro_()}:
#'       \describe{
#'         \item{\code{index}}{Index of the coefficients of interest.}
#'         \item{\code{class_index}}{Index of the class of coefficients of interest.}
#'       }
#'   }
#' }
#'
#' }
#'
#'
#' @seealso \link{cgdro_}, \link{predict_cgdro_}, \link{infer_cgdro_}, \link{summary_cgdro_}
#' @family cgdro
#' @name cgdro-family
#' @title CGDRO Family: consolidated arguments per family and complete examples
#' @keywords documentation
#'
#'
#' @example inst/examples/family_reg_ld.R
#' @example inst/examples/family_reg_hd.R
#' @examplesIf requireNamespace("xgboost", quietly = TRUE)
#' @example inst/examples/family_reg_ml.R
#' @example inst/examples/family_cls.R
NULL






#' Fit Conditional Group DRO Model in the Target Domain
#'
#' Aggregates models from multiple source domains to generate a target-domain predictor
#' using the CGDRO framework. Supports multiple problem families including low-dimensional
#' regression, high-dimensional regression, machine-learning regression, and classification.
#'
#' @param X_list List of feature matrices from each source (each \eqn{n_\ell \times p}).
#' @param y_list List of outcome vectors from each source (each \eqn{n_\ell \times 1}).
#' @param X0 Optional target feature matrix (\eqn{N \times p}); if \code{NULL}, pooled source data are used.
#' @param index Optional integer; index of the loading vector (1-based) for families that focus on a specific coordinate (only used when \code{family = "reg_hd"}).
#' @param family Character; the CGDRO family to solve. One of \code{"reg_ld"}, \code{"reg_hd"}, \code{"reg_ml"}, or \code{"cls"}.
#' @param f_learner Character; outcome-model learner. Options include \code{"linear"}, \code{"xgb"}, \code{"xgb.cv"}, \code{"high_d"}. Default is \code{"xgb"}. Not required for \code{family = "reg_ld"} or \code{"reg_hd"}.
#' @param w_learner Character; density-ratio or weight-model learner. Options include \code{"logistic"}, \code{"xgb"}, \code{"xgb.cv"}, \code{"kliep"}. Default is \code{"logistic"}. Not required for \code{family = "reg_ld"} or \code{"reg_hd"}.
#' @param loss_type Character; loss type for weight optimization. Options: \code{"reward"}, \code{"squaredloss"}, \code{"regret"}. Default is \code{"reward"}. Only needed for \code{family = "reg_ld"} or \code{"reg_ml"}.
#' @param verbose Logical; whether to print fitting progress (default: \code{FALSE}).
#' @param ... See \link{cgdro-family} for full documentation of these options.
#'
#'
#' @return A list of results including aggregated weights \code{weight_}, coefficient estimators (e.g., \code{est_}, \code{est_bc_}, \code{est_plug_}, \code{coef_}), and components required for subsequent methods such as \code{predict_cgdro_()}, \code{infer_cgdro_()}, and \code{summary_cgdro_()}.
#'
#'
# ---- Imports for CGDRO package (merged & deduped) ----
#' @importFrom glmnet glmnet cv.glmnet
#' @importFrom xgboost xgb.cv xgb.DMatrix xgb.train
#' @importFrom CVXR   Variable Problem Minimize quad_form solve is_dcp sum_entries p_norm
#' @importFrom densratio densratio
#' @importFrom nnet  multinom
#' @importFrom MASS  mvrnorm
#' @importFrom stats model.matrix predict lm qnorm cov optim rnorm qchisq
#' @importFrom utils modifyList
#' @importFrom parallel mclapply
#' @importFrom SIHR  LF


#' @seealso \link{cgdro-family}
#' @family cgdro
#' @name cgdro_
#' @title Fit Conditional Group DRO Model in the Target Domain
#' @export

cgdro_ <- function(
    X_list, y_list,
    X0 = NULL,
    index = NULL,
    family   = c("reg_ld", "reg_hd", "reg_ml", "cls"),
    f_learner = c("linear", "xgb", "xgb.cv", "high_d"),
    w_learner = c("logistic", "xgb", "xgb.cv", "ulsif"),
    loss_type = NULL,   # only used for reg_ld and reg_ml
    verbose = FALSE,
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
      loss_type = loss_type, verbose=verbose, ...
    )

  } else if (family == "reg_hd") {
    if (is.null(index)) stop("`index` must be provided for high-dimensional regression (family = 'reg_hd').")
    fit <- fit_reg_hd(
      X_list, y_list, index, X0, verbose=verbose,
      ...  # no loss_type here
    )

  } else if (family == "reg_ml") {
    fit <- fit_reg_ml(
      X_list, y_list, X0,
      loss_type = loss_type,
      f_learner = f_learner,
      w_learner = w_learner,
      verbose = verbose,
      ...
    )

  } else if (family == "cls") {
    fit <- fit_cls(
      X_list, y_list, X0,
      f_learner = f_learner,
      w_learner = w_learner,
      verbose = verbose,
      ...
    )
  }

  fit
}

#' Prediction on Target Domain using Aggregated Coefficients
#'
#' Applies the aggregated weights and coefficients returned by \code{\link{cgdro_}()}
#' to a target or user-specified design matrix, producing predicted values.
#'
#' @param fit A fitted result object returned by \code{\link{cgdro_}()}.
#' @param X   Optional numeric design matrix for the target domain. If \code{NULL},
#'            the matrix \code{fit$X0_use} defined during fitting will be used.
#' @return A numeric vector of predicted outcomes, of length equal to \code{nrow(X)}.
#' \item{pred}{Predicted labels on the target domain.}
#' \item{pred_proba}{Predicted probabilities on the target domain. Only for \code{family = "cls"}.}
#'
#' @seealso \link{cgdro-family}, \link{cgdro_}, \link{infer_cgdro_}
#' @family cgdro
#' @name predict_cgdro_
#' @title Prediction on Target Domain (CGDRO)
#' @export

predict_cgdro_ <- function(fit, X=NULL){
  if (!is.list(fit) || is.null(fit$family)) stop("fit must be a list returned by cgdro_()")
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


#' Construct confidence intervals and perform inference for fitted CGDRO models
#'
#' Builds confidence intervals and conducts statistical inference for
#' estimators obtained from \code{\link{cgdro_}()}. This function performs
#' Resampling to estimate variability and provides coordinate-wise or loading-wise
#' confidence intervals for the parameters of interest.
#'
#' @param fit A fitted CGDRO result object returned by \code{\link{cgdro_}()}.
#' @param M Integer; number of Monte Carlo samples used to construct the intervals.
#'   Default is \code{50}.
#' @param alpha Numeric; significance level for interval construction.
#'   Default is \code{0.05}.
#' @param ... See \link{cgdro-family} for full documentation of these options.
#'
#' @return A list containing:
#' \item{CI}{A numeric matrix of confidence intervals for the estimators of interest.}
#'
#' @seealso \link{cgdro-family}, \link{cgdro_}, \link{summary_cgdro_}
#' @family cgdro
#' @name infer_cgdro_
#' @title Construct confidence intervals and perform inference for fitted CGDRO models
#' @export


infer_cgdro_ <- function(fit, M = 50, alpha = 0.05, ...){
  if (!is.list(fit) || is.null(fit$family)) stop("fit must be a list returned by cgdro_()")
  if (fit$family == "reg_ld") {
    infer_reg_ld(fit, M, alpha, ...)
  } else if (fit$family == "reg_hd") {
    infer_reg_hd(fit, M, alpha, ...)
  } else if (fit$family == "cls") {
    infer_cls(fit, M, alpha, ...)
  }
}


#' Console Summary of CGDRO Results
#'
#' Displays the source‐weights, aggregated coefficients, and optionally confidence intervals
#' for a fitted CGDRO model. Provides users with a readable overview of the model output
#' and key inference results.
#'
#' @param fit   A fitted result object returned by \code{\link{cgdro_}()}.
#' @param infer Optional list returned by \code{\link{infer_cgdro_}()} containing confidence intervals.
#' @param ... See \link{cgdro-family} for full documentation of these options.
#'
#' @return Invisibly returns \code{NULL}. The function prints formatted tables to the console.
#'
#' @seealso \link{cgdro-family}, \link{cgdro_}, \link{infer_cgdro_}
#' @family cgdro
#' @name summary_cgdro_
#' @title Console Summary of CGDRO Results
#' @export

summary_cgdro_ <- function(fit, infer = NULL, ...) {
  if (!is.list(fit) || is.null(fit$family)) stop("fit must be a list returned by cgdro_()")
  if (fit$family == "reg_ld") {
    summary_reg_ld(fit, infer, ...)
  } else if (fit$family == "reg_hd") {
    summary_reg_hd(fit, infer, ...)
  } else if (fit$family == "cls") {
    summary_cls(fit, infer, ...)
  } else if (fit$family == "reg_ml") {
    print("Use predict_cgdro_() for reg_ml Instead.")
  }
}
