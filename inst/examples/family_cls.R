## Example: Classification (family = "cls")
\donttest{
  # two source groups, each with 100 samples, and 1000 target samples
  n = 100; p = 5; L = 2; N = 1000; K = 2
  data <- simu_cls(n, N, p, L, K, seed=123)
  Xlist = data$X_list
  Ylist = data$Y_list
  X0 = data$X0

  ## fit cgdro
  fit <- cgdro_(Xlist, Ylist, X0,
               family = "cls", f_learner = "linear", w_learner = "logistic")
  inf <- infer_cgdro_(fit, M = 200, alpha = 0.05, parallel = FALSE, diag = TRUE)

  ## summary
  summary_cgdro_(fit, infer = inf)

  summary_cgdro_(fit, infer = inf, index = c(1,3), class_index = c(2))

  ## prediction
  pred <- predict_cgdro_(fit)  # N x C matrix of predicted probabilities
  head(pred$pred_proba)
  head(pred$pred)
}
