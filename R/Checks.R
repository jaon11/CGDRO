check_arg_drlm_cls_fit <- function(X_list = NULL, y_list = NULL, X0 = NULL, f_learner = NULL, w_learner = NULL,
                               split = NULL, max_iter = NULL, tol = NULL, check_dual = NULL, verbose = NULL,
                               seed = NULL){
  if(is.null(X_list) || (!is.list(X_list))) stop("X_list must be a list")
  if(is.null(y_list) || (!is.list(y_list))) stop("y_list must be a list")
  if(length(X_list) != length(y_list)) stop("X_list and y_list must have the same length")
  if(length(unique(sapply(X_list, FUN=ncol))) != 1) stop("each group should have the same dimension of covariates")
  d = unique(sapply(X_list, FUN=ncol))
  if(any(sapply(X_list, nrow)!= sapply(y_list, length))) stop("The group should match in Xlist and Ylist:
                                                       they should have the same number of observations for each group")
  if(!is.null(X0)){
    if(!is.numeric(X0)|| ncol(X0)!=d) stop("X0 must be a numeric matrix with the same dimension as source covariates.")
  }
  if(is.null(f_learner) || !(f_learner %in% c("linear", "xgb", "xgb.cv"))) stop("f_learner must be one of 'linear', 'xgb', 'xgb.cv'")
  if(is.null(w_learner) || !(w_learner %in% c("linear", "xgb", "xgb.cv","kliep"))) stop("w_learner must be one of 'linear', 'xgb', 'xgb.cv','kliep'")
  if(is.null(split) || !is.logical(split)) stop("split must be TRUE or FALSE")
  if(is.null(max_iter) || !is.numeric(max_iter) || length(max_iter)!=1 || max_iter<=0 || max_iter!=round(max_iter)) stop("max_iter must be a positive integer")
  if(is.null(tol) || !is.numeric(tol) || length(tol)!=1 || tol<=0) stop("tol must be a positive numeric value")
  if(is.null(check_dual) || !is.logical(check_dual)) stop("check_dual must be TRUE or FALSE")
  if(is.null(verbose) || !is.logical(verbose)) stop("verbose must be TRUE or FALSE")
  if(is.null(seed) || !is.numeric(seed) || length(seed)!=1 || seed!=round(seed)) stop("seed must be an integer")


}


check_arg_drlm_cls_inf <- function(fit = NULL, M = NULL, alpha = NULL, parallel = NULL, n_workers = NULL, diag = NULL){
  if(is.null(fit) || !is.list(fit)) stop("fit must be a list returned by fit_drlm_cls")
  if(is.null(M) || !is.numeric(M) || length(M)!=1 || M<=0 || M!=round(M)) stop("M must be a positive integer")
  if(is.null(alpha) || !is.numeric(alpha) || length(alpha)!=1 || alpha<=0 || alpha>=1) stop("alpha must be a numeric value in (0,1)")
  if(is.null(parallel) || !is.logical(parallel)) stop("parallel must be TRUE or FALSE")
  if(is.null(n_workers) || !is.numeric(n_workers) || length(n_workers)!=1 || n_workers<=0 || n_workers!=round(n_workers)) stop("n_workers must be a positive integer")
  if(is.null(diag) || !is.logical(diag)) stop("diag must be TRUE or FALSE")
}

check_arg_drlm_reg <- function(X_list = NULL, y_list = NULL, loading_mat = NULL, X0 = NULL, intercept = NULL,
                               intercept_loading = NULL, delta = NULL, lambda = NULL, verbose = NULL){
  if(is.null(X_list) || (!is.list(X_list))) stop("X_list must be a list")
  if(is.null(y_list) || (!is.list(y_list))) stop("y_list must be a list")
  if(length(X_list) != length(y_list)) stop("X_list and y_list must have the same length")
  if(length(unique(sapply(X_list, FUN=ncol))) != 1) stop("each group should have the same dimension of covariates")
  d = unique(sapply(X_list, FUN=ncol))
  if(any(sapply(X_list, nrow)!= sapply(y_list, length))) stop("The group should match in Xlist and Ylist:
                                                       they should have the same number of observations for each group")
  if(!is.null(X0)){
    if(!is.numeric(X0)|| ncol(X0)!=d) stop("X0 must be a numeric matrix with the same dimension as source covariates.")
  }
  if(is.null(loading_mat) || !is.numeric(loading_mat)) stop("loading must be a numeric matrix")
  if(p != nrow(loading_mat)) stop("ncol(X) and nrow(loading) must match")

  if(is.null(intercept) || !is.logical(intercept)) stop("intercept must be TRUE or FALSE")
  if(is.null(intercept_loading) || !is.logical(intercept_loading)) stop("intercept_loading must be TRUE or FALSE")
  if(!is.null(delta)){
    if(!is.numeric(delta) || length(delta)!=1 || delta>0) stop("delta must be a non-positive numeric value")
  }
  if(!is.null(lambda)){
    if(!(is.character(lambda) && lambda %in% c("CV.min", "CV.1se"))){
      if(!is.numeric(lambda) || length(lambda)!=1 || lambda<0) stop("lambda must be a non-negative numeric value or 'CV.min' or 'CV.1se'")
    }
  }
  if(is.null(verbose) || !is.logical(verbose)) stop("verbose must be TRUE or FALSE")

}

check_arg_drlm_reg_inf <- function(fit = NULL, M = NULL, alpha = NULL, tau = NULL, alpha_thres = NULL, threshold = NULL, delta = NULL){
  if(is.null(fit) || !is.list(fit)) stop("fit must be a list returned by fit_drlm_reg")
  if(is.null(M) || !is.numeric(M) || length(M)!=1 || M<=0 || M!=round(M)) stop("M must be a positive integer")
  if(is.null(alpha) || !is.numeric(alpha) || length(alpha)!=1 || alpha<=0 || alpha>=1) stop("alpha must be a numeric value in (0,1)")
  if(is.null(tau) || !is.numeric(tau) || length(tau)!=1 || tau<=0) stop("tau must be a positive numeric value")
  if(is.null(alpha_thres) || !is.numeric(alpha_thres) || length(alpha_thres)!=1 || alpha_thres<=0 || alpha_thres>=1) stop("alpha_thres must be a numeric value in (0,1)")
  if(is.null(threshold) || !is.numeric(threshold) || length(threshold)!=1) stop("threshold must be a numeric value")
  if(!is.null(delta)){
    if(!is.numeric(delta) || length(delta)!=1 || delta>0) stop("delta must be a non-positive numeric value")
  }
}

check_arg_drol_fit <- function(X_list = NULL, y_list = NULL, X0 = NULL,
                          f_learner = NULL, w_learner = NULL,
                          bias_correct = NULL, priors = NULL,
                          ridge = NULL, solver = NULL,
                          seed = NULL){
  if(is.null(X_list) || (!is.list(X_list))) stop("X_list must be a list")
  if(is.null(y_list) || (!is.list(y_list))) stop("y_list must be a list")
  if(length(X_list) != length(y_list)) stop("X_list and y_list must have the same length")
  if(length(unique(sapply(X_list, FUN=ncol))) != 1) stop("each group should have the same dimension of covariates")
  d = unique(sapply(X_list, FUN=ncol))
  if(any(sapply(X_list, nrow)!= sapply(y_list, length))) stop("The group should match in Xlist and Ylist:
                                                       they should have the same number of observations for each group")
  if(!is.null(X0)){
    if(!is.numeric(X0)|| ncol(X0)!=d) stop("X0 must be a numeric matrix with the same dimension as source covariates.")
  }
  if(is.null(f_learner) || !(f_learner %in% c("linear", "xgb", "xgb.cv"))) stop("f_learner must be one of 'linear', 'xgb', 'xgb.cv'")
  if(is.null(w_learner) || !(w_learner %in% c("linear", "xgb", "xgb.cv","kliep"))) stop("w_learner must be one of 'linear', 'xgb', 'xgb.cv','kliep'")
  if(is.null(bias_correct) || !is.logical(bias_correct)) stop("bias_correct must be TRUE or FALSE")
  if(!is.null(priors)){
    if(!is.list(priors) || length(priors)!=2) stop("priors must be a list of length 2")
    if(!is.numeric(priors[[1]]) || any(priors[[1]]<0)) stop("the first element of priors must be a non-negative numeric vector")
    if(abs(sum(priors[[1]])-1)>1e-6) stop("the sum of the first element of priors must be 1")
    if(!is.numeric(priors[[2]]) || length(priors[[2]])!=1 || priors[[2]]<0) stop("the second element of priors must be a non-negative numeric value")
    if(length(priors[[1]]) != fit$L) stop(paste0("the length of the first element of priors must equal to the number of sources L=", fit$L))
  }
  if(is.null(ridge) || !is.numeric(ridge) || length(ridge)!=1 || ridge<0) stop("ridge must be a non-negative numeric value")
  if(is.null(solver) || !(any(solver %in% c("ECOS", "SCS")))) stop("solver must be one of 'ECOS' or 'SCS'")
  if(is.null(seed) || !is.numeric(seed) || length(seed)!=1 || seed!=round(seed)) stop("seed must be an integer")


}


