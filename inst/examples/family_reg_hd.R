## Example: High-dimensional Linear Regression (family = "reg_hd")
\donttest{
  # two source groups, each with 100 samples, and 100 target samples
  source(data/data.R)
  data <- simu_linear_reg_highd(n_list = c(100, 100), N = 100, p = 100, seed = 123)
  Xlist = data$X_list
  Ylist = data$Y_list
  X0 = data$X0

  ## fit cgdro
  fit <- cgdro_(Xlist, Ylist, X0 = X0,
               family = "reg_hd",
               index = c(1,10,45,99), intercept = FALSE,
               delta = 0, lambda = "CV.min", verbose = FALSE)
  inf <- infer_cgdro_(fit, M = 200, alpha = 0.05)

  ## summary
  summary_cgdro_(fit, infer=inf)

  ## predict
  pred <- predict_cgdro_(fit)  # N x 1 vector of predicted values
  head(pred)
}
