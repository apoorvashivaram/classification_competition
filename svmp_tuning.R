# Support Vector Machine Polynomial tuning ----

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# seed
set.seed(46291)

# load required objects ----
load("model_info/loan_setup.rda")

# define model ----
svmp_model <-
  # specify model type and parameters to optimize
  svm_poly(cost = tune(),
           degree = tune(),
           scale_factor = tune()) %>% 
  # set underlying engine/package
  set_engine("kernlab") %>% 
  # set mode
  set_mode("classification")

# check tuning parameters
svmp_params <- parameters(svmp_model)

# define tuning grid ---- 
svmp_grid <- grid_regular(svmp_params, levels = 5)

# workflow ----
svmp_workflow <- workflow() %>% 
  add_model(svmp_model) %>% 
  add_recipe(loan_recipe)

# tuning/fitting ----
tic("SVM Polynomial")

svmp_tune <- svmp_workflow %>% 
  tune_grid(
    resamples = loan_fold, 
    grid = svmp_grid
    )

toc(log = TRUE)

# save runtime info
svmp_runtime <- tic.log(format = TRUE)

# write out results & workflow
save(svmp_tune, svmp_workflow, svmp_runtime, file = "model_info/svmp_tune.rda")

