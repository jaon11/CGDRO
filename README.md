
# CGDRO

CGDRO is an R package providing conprehensive multi-source prediction
and model statistical inference without knowing target labels, offering
multi-source solutions to low-dimensional and high-dimensional, linear
and complex data, regression and classification problems.

CGDRO spports 4 familys of models:

| Family | Description | Statistical Inference |
|-----------------------------|---------------------------------|----------------|
| `reg_ld` | Linear prediction model (low-dimensional) | ✅ |
| `reg_hd` | High-dimensional linear model | ✅ |
| `reg_ml` | Machine learning prediction model | ❌ |
| `cls` | Linear model for classification task | ✅ |

# Installation

The development version from [GitHub](https://github.com) with:

``` r
# install.packages("devtools")
devtools::install_github("jaon11/CGDRO")
```

# Example

This is a basic example which shows you how to solve a common problem:

``` r
library(CGDRO)
```

The data is heterogeneous and covariates shift between source and target
data

``` r
# number of source groups = 3, with 1000 samples each
  # sigma: source group 1,3: 0.5; source group 2: 2
  # target sample size = 10000
  # dimension p = 5
  data <- simu_linear_reg_lowd(n_list = list(1000,1000,1000), N=10000, p = 5, seed = 123)
  Xlist = data$X_list
  Ylist = data$Y_list
  X0 = data$X0
```

Fitting CGDRO model for low-dimensional linear regression with reward
loss and summarize results, more loss types can be selected from `squredloss` and `regret`, showing in tutorial of [family='reg_ld'](reg_ld.ipynb).


``` r
  ## fit cgdro
  ## Note: only when loss_type='reward', infer() can be called to get confidence intervals
  ## For other loss_type, only point estimation and prediction can be done
  fit <- cgdro_(Xlist, Ylist, X0, loss_type = "reward",
               family = "reg_ld",  intercept = TRUE,
               delta = 0,  verbose = FALSE)
  inf <- infer_cgdro_(fit, M = 200, alpha = 0.05)
  ## summary
  summary_cgdro_(fit, infer=inf)
```

    ## Model Summary:
    ## =================================
    ## CGDRO Aggregated Weights:
    ## 
    ## group     |        1        2        3
    ## weight_   |   0.5523   0.2813   0.1665
    ## 
    ## =================================
    ## CGDRO Aggregated Estimators:
    ## 
    ## index     |        1        2        3        4        5        6
    ## coef_     |   0.0232  -0.0653  -0.0449   0.0333  -0.0104   0.1307
    ## 
    ## =================================
    ## Confidence Intervals:
    ## 
    ## index     |              1              2              3              4              5
    ## CI        | (-0.0467,0.1067) (-0.2273,0.0549) (-0.1714,0.0781) (-0.1005,0.1320) (-0.1502,0.1224)
    ## index     |              6
    ## CI        | (0.0247,0.2149)

We can get statistical inference results from CGDRO, including **CGDRO Aggregated Weights** (learned weights from each group of source domain), **Coefficient Estimators** (the worst-case estimators of coefficient on target domain), and **Confidence Intervals** (valid confidence intervals of target domain coefficient estimators). In the summarized results above, `group` refers to each group of source domains, `index` refers to the index of coeffients, starting from the intercept if `intercept=TRUE`, else starting from the first dimension of coefficient.

Make prediction on target data (you do not have to state the coveriate you use for prediction since target data is the default choice) and show the first 6 predicted values.


``` r
  ## predict
  pred <- predict_cgdro_(fit)  # N x 1 vector of predicted values
  head(pred)
```

    ## [1] -0.10216265 -0.23954467  0.05219648 -0.14650883 -0.02947897  0.01311665

# References

<span id="ref-guo2024statistical"></span>
**Guo, Z.** (2024). *Statistical inference for maximin effects: Identifying stable associations across multiple studies.*  
*Journal of the American Statistical Association*, 119(547), 1968–1984.  
[Paper link](https://doi.org/10.1080/01621459.2023.2258469)

<span id="ref-wang2023distributionally"></span>
**Wang, Z.**, **Bühlmann, P.**, & **Guo, Z.** (2023). *Distributionally robust machine learning with multi-source data.*  
*arXiv preprint* [arXiv:2309.02211](https://arxiv.org/abs/2309.02211)

<span id="ref-guo2025statistical"></span>
**Guo, Z.**, **Wang, Z.**, **Hu, Y.**, & **Bach, F.** (2025). *Statistical Inference for Conditional Group Distributionally Robust Optimization with Cross-Entropy Loss.*  
*arXiv preprint* [arXiv:2507.09905](https://arxiv.org/abs/2507.09905)

