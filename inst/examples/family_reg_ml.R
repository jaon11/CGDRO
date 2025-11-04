## Example: Machine-learning Regression (family = "reg_ml")
\donttest{
  # number of source groups = 3, each with 1000 samples, and 10000 target samples
  # dimension p = 5
  data <- simu_reg_ml(n_vec = c(1000,1000,1000), n0=10000, N_label=20, p=5, seed = 123)
  Xlist = data$X_list
  Ylist = data$Y_list
  X0 = data$X0

  ## fit cgdro
  fit <- cgdro_(Xlist, Ylist, X0, loss_type = "reward",
               family = "reg_ml", f_learner = "xgb", w_learner = "linear",
               bias_correct = TRUE,
               priors = NULL,
               ridge = 1e-8,
               seed = 123,
               verbose = FALSE)
  fit$weight_

  ## predict
  pred <- predict_cgdro_(fit)  # N x 1 vector of predicted values
  head(pred)

  ## fit cgdro
  fit <- cgdro_(Xlist, Ylist, X0, loss_type = "squaredloss",
               family = "reg_ml", f_learner = "xgb", w_learner = "linear",
               bias_correct = TRUE,
               priors = NULL,
               ridge = 1e-8,
               seed = 123)
  fit$weight_

  ## predict
  pred <- predict_cgdro_(fit)  # N x 1 vector of predicted values
  head(pred)

  ## fit cgdro
  fit <- cgdro_(Xlist, Ylist, X0, loss_type = "regret",
               family = "reg_ml", f_learner = "xgb", w_learner = "linear",
               bias_correct = TRUE,
               priors = NULL,
               ridge = 1e-8,
               seed = 123)
  fit$weight_

  ## predict
  pred <- predict_cgdro_(fit)  # N x 1 vector of predicted values
  head(pred)
}
