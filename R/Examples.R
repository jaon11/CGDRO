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

fit <- cgdro(X_list, y_list, X0,
             family = "drlm_cls", f_learner = "xgb", w_learner = "kliep")
fit$coef_
fit$weight

predict(fit)

inf <- infer(fit, M = 50, alpha = 0.05, parallel = FALSE, n_workers = 2, diag = TRUE)
inf$CI

summary(fit, infer=inf, dim_search = c(1,3,5), class_search = c(2))

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


fit <- cgdro(list(X1,X2), list(Y1,Y2), X0 = X0,
             family = "drlm_reg", f_learner = "high_d", w_learner = "linear",
             loading_mat = loading_mat, intercept = FALSE,
             delta = 0, lambda = "CV.min", verbose = FALSE)



inf <- infer(fit, M = 50)
inf$CI

summary(fit, infer=inf)

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
fit <- cgdro(
  Xlist, Ylist, X0,
  family = "drlm_reg", f_learner = "linear", w_learner = "linear",
  loading_mat = t(loading_mat), intercept = FALSE,
  delta = 0, lambda = "CV.min", verbose = TRUE
)

cat("\nOptimal weights across sources:\n")
print(round(fit$weight_, 4))

cat("\nEstimated loading coefficients:\n")
print(round(fit$loading_coef_, 4))

## prediction
predict(fit)

## ------------------- Inference -------------------
inf <- infer(fit, M = 50)
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

fit <- cgdro(Xlist, Ylist, X0,
             family = "drol", f_learner = "linear", w_learner = "linear", bias_correct = TRUE, priors=NULL, seed = 123)
res <- predict(fit)
fit$weight                                           # optimal weights
head(res$pred)

