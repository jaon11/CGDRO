######################### Examples to Test CGDRO #################################
###########################################################################################
#source("Main.R")
######################################################################
######################### DRlm-Classification ########################
######################################################################
# =====================================================================
# Example: Your DGP (K = 2 => C = 3)
# =====================================================================
set.seed(1234)
L <- 3; n <- 200; p <- 10; N <- 2000; K <- 2; C <- K + 1
X_list <- replicate(L, matrix(rnorm(n * p, 0, 1), n, p), simplify = FALSE)
X0 <- matrix(rnorm(N * p, 0.1, 1), N, p)
beta_list <- replicate(L, matrix(rnorm(p
                                       * K, 0, 0.25), p, K), simplify = FALSE)
logits_list <- Map(function(X, B) sweep(X %*% B, 2, colMeans(X %*% B), "-"), X_list, beta_list)
probsK_list <- lapply(logits_list, .softmax_reduced)
probs_list <- lapply(probsK_list, function(PK) cbind(1 - rowSums(PK), PK))
y_list <- lapply(probs_list, function(Pr) apply(Pr, 1, function(pr) which.max(rmultinom(1, 1, pr)) - 1))

fit <- fit_drlm_cls(X_list, y_list, X0,
                    prob_learner = "linear", density_learner = "linear",
                    split = TRUE, intercept = FALSE, max_iter = 300, tol = 1e-5, verbose = TRUE, seed = 123)
head(predict_proba_drlm_cls(fit))
inf <- infer_drlm_cls(fit, index = 1, M = 50, alpha = 0.05, parallel = FALSE, diag = TRUE)
inf$CI_index

inf$CI_U # list of K matrices (d, 2)

fit$theta # d*K matrix

fit$gamma

inf$CI_lb_U # d*K matrix
inf$CI_ub_U # d*K matrix


######################################################################
########################### DRlm-Regression ##########################
######################################################################
# --------------------------- example (optional) ---------------------------
set.seed(0)
p <- 100; L <- 2
A1gen <- function(rho, p) { i <- matrix(rep(1:p, each=p), p); j <- t(i); rho^abs(i - j) }
cov_source <- A1gen(0.6, p)
X1 <- MASS::mvrnorm(100, mu = rep(0,p), Sigma = cov_source)
X2 <- MASS::mvrnorm(100, mu = rep(0,p), Sigma = cov_source)
b1 <- rep(0,p); b1[1:5] <- (1:5)/20; b1[97:99] <- c(0.5, -0.5, -0.5)
b2 <- rep(0,p); b2[6:10] <- (1:5)/20; b2[97:99] <- 0.5*c(0.5,-0.5,-0.5)
Y1 <- as.numeric(X1 %*% b1 + rnorm(100))
Y2 <- as.numeric(X2 %*% b2 + rnorm(100))
cov0 <- cov_source; diag(cov0) <- 1.5; cov0[1:5,1:5] <- 0.9; diag(cov0[1:5,1:5]) <- 1.5; cov0[99:100,99:100] <- matrix(c(1.5,0.9,0.9,1.5),2)
X0 <- MASS::mvrnorm(100, mu = rep(0,p), Sigma = cov0)
loading_mat <- matrix(0, nrow = 100, ncol = 2); loading_mat[96:100,1] <- 0.4; loading_mat[99:100,2] <- 0.8 #; loading_mat <- t(loading_mat)
#
fit <- fit_drlm_reg(list(X1,X2), list(Y1,Y2), loading_mat, X0 = X0, intercept = FALSE,
                    delta = 0, lambda = "CV.min", verbose = TRUE)
print(fit$weight_)
print(fit$loading_coef_)
inf <- infer_drlm_reg(fit, M = 50)
print(inf$CI)


# ------------------- Example: p=5, L=2 -------------------

set.seed(0)
L <- 2
p <- 5

# Source distributions
mean_source <- rep(0, p)
cov_source  <- diag(p)

# --- Group 1 ---
n1 <- 100
X1 <- MASS::mvrnorm(n1, mu = mean_source, Sigma = cov_source)
b1 <- rep(0, p); b1[1:5] <- (1:5) / 20
Y1 <- as.numeric(X1 %*% b1 + rnorm(n1))

# --- Group 2 ---
n2 <- 100
X2 <- MASS::mvrnorm(n2, mu = mean_source, Sigma = cov_source)
b2 <- rev(2:6) / 20   # same as np.arange(6,1,-1)/20
Y2 <- as.numeric(X2 %*% b2 + rnorm(n2))

# --- Target (covariate shift: here same cov as sources) ---
n0 <- 100
X0 <- MASS::mvrnorm(n0, mu = rep(0, p), Sigma = cov_source)

Xlist <- list(X1, X2)
Ylist <- list(Y1, Y2)

# Define a simple loading matrix: pick first 2 coordinates
loading_mat <- diag(p)[1:2, ]

## ------------------- Fit DRO regression -------------------
fit <- fit_drlm_reg(
  X_list = Xlist, y_list = Ylist,
  loading_mat = loading_mat,
  X0 = X0,
  intercept = FALSE,
  delta = 0,
  lambda = "CV.min",
  verbose = TRUE
)

cat("\nOptimal weights across sources:\n")
print(round(fit$weight_, 4))

cat("\nEstimated loading coefficients:\n")
print(round(fit$loading_coef_, 4))

## prediction
predict_drlm_reg(fit)

## ------------------- Inference -------------------
inf <- infer_drlm_reg(
  fit, M = 50, alpha = 0.05,
  tau = 0.2, alpha_thres = 0.01, threshold = 0
)

cat("\nUnion 95% CIs for loadings:\n")
print(round(inf$CI, 4))







######################################################################
################################ DRoL ################################
######################################################################


set.seed(0)
L <- 2; p <- 5
mean_source <- rep(0, p); cov_source <- diag(p)
n1 <- 2000; n2 <- 2000; n0 <- 20000
X1 <- MASS::mvrnorm(n1, mean_source, cov_source)
X2 <- MASS::mvrnorm(n2, mean_source, cov_source)
b1 <- c(1,-1,0.5,0,0) + rnorm(p,0,0.1)
b2 <- c(1,0.5,-0.5,0,0) + rnorm(p,0,0.1)
Y1 <- (X1^3 %*% b1) + sin(X1 %*% b1) + pmin(pmax(exp(X1 %*% b1),0),1) + rnorm(n1)
Y2 <- (X2^3 %*% b2) + sin(X2 %*% b2) + pmin(pmax(exp(X2 %*% b2),0),1) + rnorm(n2)
X0 <- MASS::mvrnorm(n0, rep(0,p), cov_source)
Xlist <- list(X1, X2); Ylist <- list(Y1, Y2)

fit <- fit_drol(Xlist, Ylist, X0,
                outcome_learner = "xgb",
                density_learner = "linear",
                intercept = FALSE, seed = 1234)
res <- predict_drol(fit, bias_correct = TRUE)         # CVXR optimization
res$weight_                                           # optimal weights
head(res$pred)

