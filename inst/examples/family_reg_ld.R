## Example: Low-dimensional Linear Regression (family = "reg_ld")
\donttest{
  # number of source groups = 3, with 1000 samples each
  # sigma: source group 1,3: 0.5; source group 2: 2
  # target sample size = 10000
  # dimension p = 5
  data <- simu_linear_reg_lowd(n_list = list(1000,1000,1000), N=10000, p = 5, seed = 123)
  Xlist = data$X_list
  Ylist = data$Y_list
  X0 = data$X0
  ## fit cgdro
  ## Note: only when loss_type='reward', infer() can be called to get confidence intervals
  ## For other loss_type, only point estimation and prediction can be done
  fit <- cgdro_(Xlist, Ylist, X0, loss_type = "reward",
               family = "reg_ld",  intercept = TRUE,
               delta = 0,  verbose = FALSE)
  inf <- infer_cgdro_(fit, M = 200, alpha = 0.05)
  ## summary
  summary_cgdro_(fit, infer=inf)
  ## predict
  pred <- predict_cgdro_(fit)  # N x 1 vector of predicted values
  head(pred)


  ## fit cgdro
  fit <- cgdro_(Xlist, Ylist, X0, loss_type = "squaredloss",
               family = "reg_ld",  intercept = TRUE,
               delta = 0,  verbose = FALSE)
  ## summary
  summary(fit)
  ## predict
  pred <- predict_cgdro_(fit)  # N x 1 vector of predicted values
  head(pred)


  ## fit cgdro
  fit <- cgdro_(Xlist, Ylist, X0, loss_type = "regret",
               family = "reg_ld",  intercept = TRUE,
               delta = 0,  verbose = FALSE)
  ## summary
  summary(fit)
  ## predict
  pred <- predict_cgdro_(fit)  # N x 1 vector of predicted values
  head(pred)

}
