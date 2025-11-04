library(MASS)

# ---- Helpers ----
softmax_mat <- function(X) {
  # row-wise softmax for a matrix
  Xmax <- apply(X, 1, max)
  Exp  <- exp(X - Xmax)
  Exp / rowSums(Exp)
}

AR1_cov <- function(rho, p) {
  i <- 0:(p-1)
  absdiff <- abs(outer(i, i, "-"))
  rho^absdiff
}

# ===========================================================
# 1) Linear regression, low dimension 
# ===========================================================
simu_linear_reg_lowd <- function(n_list, N, p = 5, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  L <- length(n_list)
  
  mean_source <- rep(0, p)
  cov_source  <- diag(p)
  mean_target <- rep(0, p)
  cov_target  <- diag(p)
  
  # beta list (exactly like Python)
  b1 <- rep(0, p); b1[1:5] <- (1:5) / 20
  b2 <- -rev(2:6) / 15  # seq 6,5,4,3,2 divided by 15 with minus sign
  b3 <- c(0.175, -0.175, 0.175, -0.175, 0.175)
  beta_list <- list(b1, b2, b3)[1:L]
  
  # Sources
  X_list <- vector("list", L)
  Y_list <- vector("list", L)
  for (l in seq_len(L)) {
    n <- n_list[[l]]
    X <- MASS::mvrnorm(n = n, mu = mean_source, Sigma = cov_source)
    eps_sd <- if (l == 2) 2 else 0.5
    Y <- as.vector(X %*% beta_list[[l]] + rnorm(n, sd = eps_sd))
    X_list[[l]] <- X
    Y_list[[l]] <- Y
  }
  
  # Target covariates
  X0 <- MASS::mvrnorm(n = N, mu = mean_target, Sigma = cov_target)
  
  list(
    X_list  = X_list,
    Y_list  = Y_list,
    F_list  = beta_list,  # parameter list
    X0      = X0,
    meta    = list(mean_source = mean_source, cov_source = cov_source,
                   mean_target = mean_target, cov_target = cov_target)
  )
}

# ===========================================================
# 2) Linear regression, high dimension
# ===========================================================
simu_linear_reg_highd <- function(n_list, N, p = 100, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  L <- length(n_list)
  
  mean_source <- rep(0, p)
  cov_source  <- AR1_cov(rho = 0.6, p = p)
  
  mean_target <- rep(0.1, p)
  cov_target  <- cov_source
  diag(cov_target) <- 1.5
  # Strong within-block correlations:
  # first 5 features
  for (i in 1:5) for (j in 1:5) if (i != j) cov_target[i, j] <- 0.9
  # features 98:100 (R indices)
  for (i in 98:100) for (j in 98:100) if (i != j) cov_target[i, j] <- 0.9
  
  # beta list
  b1 <- rep(0, p); b1[1:5] <- (1:5) / 20; b1[98:100] <- c(0.5, -0.5, -0.5) # guard
  b1[97:99] <- c(0.5, -0.5, -0.5)
  b2 <- rep(0, p); b2[6:10] <- (1:5) / 20; b2[98:100] <- 0.5 * c(0.5, -0.5, -0.5)
  beta_list <- list(b1, b2)[1:L]
  
  # Sources
  X_list <- vector("list", L)
  Y_list <- vector("list", L)
  for (l in seq_len(L)) {
    n <- n_list[[l]]
    X <- MASS::mvrnorm(n = n, mu = mean_source, Sigma = cov_source)
    Y <- as.vector(X %*% beta_list[[l]] + rnorm(n))
    X_list[[l]] <- X
    Y_list[[l]] <- Y
  }
  
  # Target covariates
  X0 <- MASS::mvrnorm(n = N, mu = mean_target, Sigma = cov_target)
  
  list(
    X_list  = X_list,
    Y_list  = Y_list,
    F_list  = beta_list,  # parameter list
    X0      = X0,
    meta    = list(mean_source = mean_source, cov_source = cov_source,
                   mean_target = mean_target, cov_target = cov_target)
  )
}

# ===========================================================
# 3) ML regression
#    - Returns potential outcomes on target as well
# ===========================================================
simu_reg_ml <- function(n_vec, n0, N_label, p = 5, seed = NULL) {
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("Please install.packages('MASS') to use simu_reg_ml().")
  }
  if (!is.null(seed)) set.seed(seed)
  stopifnot(length(n_vec) == 3)

  mean_source <- rep(0, p)
  cov_source  <- diag(p)

  n1 <- n_vec[1]; n2 <- n_vec[2]; n3 <- n_vec[3]

  X1 <- MASS::mvrnorm(n1, mean_source, cov_source)
  X2 <- MASS::mvrnorm(n2, mean_source, cov_source)
  X3 <- MASS::mvrnorm(n3, mean_source, cov_source)

  # Coefficients (your intended three distinct b's)
  b1 <- c(1, -1, 0.5, 0.5, 1)       + rnorm(p, 0, 0.1)
  b2 <- c(1,  0.5, -0.5, 0.5, -1)    + rnorm(p, 0, 0.1)
  b3 <- c(1, -0.5, -1, -0.5, 0.5)     + rnorm(p, 0, 0.1)

  beta_list <- list(b1, b2, b3)

  # elementwise cube, then %*% b
  f_term <- function(X, b) {
    (X^3 %*% b) + sin(X %*% b) + pmin(pmax(exp(X %*% b), 0), 1)
  }

  Y1 <- as.vector(f_term(X1, b1) + rnorm(n1) * 0.5)
  Y2 <- as.vector(f_term(X2, b2) + rnorm(n2) * 7)
  Y3 <- as.vector(f_term(X3, b3) + rnorm(n3) * 0.5)

  X0 <- MASS::mvrnorm(n0, rep(0, p), cov_source)

  # Faithful to Python: weights = c(0.6, rep(0.4, L-1))  (note: not normalized unless L==2)
  weights <- c(0.6, rep(0.4, 3 - 1))

  Y0 <- rnorm(n0)
  for (l in seq_len(3)) {
    Y0 <- Y0 + weights[l] * f_term(X0, beta_list[[l]])
  }

  idx_lab <- seq_len(N_label)
  X0_label <- X0[idx_lab, , drop = FALSE]
  Y0_label <- Y0[idx_lab]



  list(
    X_list    = list(X1, X2, X3),
    Y_list    = list(Y1, Y2, Y3),
    beta_list = list(b1, b2, b3),
    X0        = X0,
    Y0        = as.numeric(Y0),
    X0_label  = X0_label,
    Y0_label  = as.numeric(Y0_label)
  )
}



# install.packages("MASS")
library(MASS)

# Helper: row-wise quadratic form
.row_quad <- function(X, A) {
  XA <- X %*% A
  rowSums(XA * X)
}

# Helper: build centered f_funcs list (shared by both DGPs)
.build_centered_funcs <- function(L, d, mu0, seed_funcs = NULL) {
  if (!is.null(seed_funcs)) set.seed(seed_funcs)
  # sample from target covariate distribution for centering
  X_sample <- matrix(rnorm(20000 * d), ncol = d)
  X_sample <- sweep(X_sample, 2, mu0, "+")

  f_list <- vector("list", L)
  for (l in seq_len(L)) {
    beta <- runif(d, min = -1, max = 1)
    B <- matrix(runif(d * d, min = -0.5, max = 0.5), nrow = d)
    A <- (B + t(B)) / 2
    c_const <- sum(diag(A)) + as.numeric(t(mu0) %*% A %*% mu0)

    f_raw <- function(X) {
      as.numeric(sin(X %*% beta)) + .row_quad(X, A) - c_const
    }
    shift <- mean(f_raw(X_sample))

    # capture by value
    f_centered <- local({
      f0 <- f_raw; sh <- shift
      function(X) f0(X) - sh
    })
    f_list[[l]] <- f_centered
  }
  f_list
}

# =========================================================
# DGP 1: Nonlinear_reg (potential outcomes per source)
# =========================================================
simulate_nonlinear_reg <- function(
  n, N, L,
  seed_funcs = NULL,
  seed_data  = NULL
) {
  d <- 5
  mu0 <- c(1, -1, 0.5, 0, 0)
  Sigma0 <- diag(d)

  # build centered functions
  f_funcs <- .build_centered_funcs(L, d, mu0, seed_funcs)

  if (!is.null(seed_data)) set.seed(seed_data)

  # -------- Source data --------
  mu <- rep(0, d); Sigma <- diag(d)
  X_sources_list <- vector("list", L)
  Y_sources_list <- vector("list", L)
  for (l in seq_len(L)) {
    X <- MASS::mvrnorm(n, mu = mu, Sigma = Sigma)
    # Python used l==1 (0-based) for larger noise; here it's l==2 (1-based)
    sd_eps <- if (l == 2) 3 else 0.5
    Y <- f_funcs[[l]](X) + rnorm(n, sd = sd_eps)
    X_sources_list[[l]] <- X
    Y_sources_list[[l]] <- as.numeric(Y)
  }

  # -------- Target data --------
  X_target <- MASS::mvrnorm(N, mu = mu0, Sigma = Sigma0)
  Y_target_potential_list <- vector("list", L)
  for (l in seq_len(L)) {
    Yt <- f_funcs[[l]](X_target) + rnorm(N)
    Y_target_potential_list[[l]] <- as.numeric(Yt)
  }

  list(
    n = n, N = N, d = d, L = L,
    mu0 = mu0, Sigma0 = Sigma0,
    f_funcs = f_funcs,
    X_sources_list = X_sources_list,
    Y_sources_list = Y_sources_list,
    X_target = X_target,
    Y_target_potential_list = Y_target_potential_list
  )
}

# =========================================================
# DGP 2: Nonlinear_reg_prior (mixture target with labels)
# =========================================================
simulate_nonlinear_reg_prior <- function(
  n, N, N_label,
  L = 4,
  seed_funcs = NULL,
  seed_data  = NULL
) {
  d <- 5
  mu0 <- c(1, -1, 0.5, 0, 0)
  Sigma0 <- diag(d)

  # build centered functions
  f_funcs <- .build_centered_funcs(L, d, mu0, seed_funcs)

  if (!is.null(seed_data)) set.seed(seed_data)

  # -------- Source data --------
  mu <- rep(0, d); Sigma <- diag(d)
  X_sources_list <- vector("list", L)
  Y_sources_list <- vector("list", L)
  for (l in seq_len(L)) {
    X <- MASS::mvrnorm(n, mu = mu, Sigma = Sigma)
    Y <- f_funcs[[l]](X) + rnorm(n)
    X_sources_list[[l]] <- X
    Y_sources_list[[l]] <- as.numeric(Y)
  }

  # -------- Target data (mixture of f_l) --------
  X_target <- MASS::mvrnorm(N, mu = mu0, Sigma = Sigma0)

  # Faithful to Python: weights = c(0.6, rep(0.4, L-1))  (note: not normalized unless L==2)
  weights <- c(0.6, rep(0.4, L - 1))

  Y_target <- rnorm(N)
  for (l in seq_len(L)) {
    Y_target <- Y_target + weights[l] * f_funcs[[l]](X_target)
  }

  idx_lab <- seq_len(N_label)
  X_target_label <- X_target[idx_lab, , drop = FALSE]
  Y_target_label <- Y_target[idx_lab]

  list(
    n = n, N = N, N_label = N_label, d = d, L = L,
    mu0 = mu0, Sigma0 = Sigma0,
    f_funcs = f_funcs,
    X_sources_list = X_sources_list,
    Y_sources_list = Y_sources_list,
    X_target = X_target,
    Y_target = as.numeric(Y_target),
    X_target_label = X_target_label,
    Y_target_label = as.numeric(Y_target_label)
  )
}

# -------------------------
# Minimal usage examples
# -------------------------
# dgp1 <- simulate_nonlinear_reg(n = 500, N = 2000, L = 4, seed_funcs = 123, seed_data = 42)
# str(dgp1$X_sources_list[[1]]); length(dgp1$Y_target_potential_list)

# dgp2 <- simulate_nonlinear_reg_prior(n = 500, N = 2000, N_label = 200, L = 4,
#                                      seed_funcs = 123, seed_data = 42)
# head(dgp2$Y_target_label)




# ===========================================================
# 4) Linear classification (multinomial)
#    NOTE: Following Python exactly:
#          beta has (K+1) columns: one all-zeros "baseline" + K random columns
#          => the number of classes is (K+1).
# ===========================================================
simu_cls <- function(n, N, p, L, K, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  # Build beta_list: each is p x (K+1)
  beta_list <- vector("list", L)
  for (l in seq_len(L)) {
    beta <- cbind(
      matrix(0, nrow = p, ncol = 1),
      matrix(rnorm(p * K, mean = 0, sd = 0.25), nrow = p, ncol = K)
    )
    beta_list[[l]] <- beta
  }
  
  # Sources
  X_list    <- vector("list", L)
  Y_list    <- vector("list", L)
  probs_list <- vector("list", L)
  for (l in seq_len(L)) {
    X <- matrix(rnorm(n * p), nrow = n, ncol = p)
    logits <- X %*% beta_list[[l]]
    logits <- logits - mean(logits) # center
    P <- softmax_mat(logits)
    # sample class in {0,...,K} (K+1 classes total)
    y <- apply(P, 1, function(pr) {
      which(rmultinom(1, size = 1, prob = pr) == 1) - 1L
    })
    X_list[[l]]     <- X
    Y_list[[l]]     <- y
    probs_list[[l]] <- P
  }
  
  # Target domain (covariate shift only)
  X0 <- matrix(rnorm(N * p, mean = 0.1, sd = 1), nrow = N, ncol = p)
  
  list(
    X_list     = X_list,
    Y_list     = Y_list,
    F_list     = beta_list,   # parameter list (beta matrices)
    X0         = X0,
    probs_list = probs_list
  )
}
