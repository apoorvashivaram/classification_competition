# Support Vector Machine Radial Basis Function tuning ----

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# seed
set.seed(75326)

# load required objects ----
load("model_info/loan_setup.rda")

# define model ----
svm_rbf_model <-
  # specify model type and parameters to optimize
  svm_rbf(cost = tune(),
          rbf_sigma = tune()) %>% 
  # set underlying engine/package
  set_engine("kernlab") %>% 
  # set mode
  set_mode("classification")

# check tuning parameters
svm_rbf_params <- parameters(svm_rbf_model)

# define tuning grid ---- 
svm_rbf_grid <- grid_regular(svm_rbf_params, levels = 5)

# workflow ----
svm_rbf_workflow <- workflow() %>% 
  add_model(svm_rbf_model) %>% 
  add_recipe(loan_recipe)

# tuning/fitting ----
tic("SVM RBF")

svm_rbf_tune <- svm_rbf_workflow %>% 
  tune_grid(
    resamples = loan_fold, 
    grid = svm_rbf_grid
    )

toc(log = TRUE)

# save runtime info
svm_rbf_runtime <- tic.log(format = TRUE)

# write out results & workflow
save(svm_rbf_tune, svm_rbf_workflow, svm_rbf_runtime, file = "model_info/svm_rbf_tune.rda")

