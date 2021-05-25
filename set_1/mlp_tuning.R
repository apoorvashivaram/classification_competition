# single Layer Neural Network (multilayer perceptron --- mlp) tuning ----

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# seed
set.seed(32213)

# load required objects ----
load("model_info/loan_setup.rda")

# define model ----
mlp_model <-
  # specify model type and parameters to optimize
  mlp(hidden_units = tune(),
      penalty = tune()) %>% 
  # set underlying engine/package
  set_engine("nnet") %>% 
  # set mode
  set_mode("classification")

# check tuning parameters
mlp_params <- parameters(mlp_model)

# define tuning grid ---- 
mlp_grid <- grid_regular(mlp_params, levels = 5)

# workflow ----
mlp_workflow <- workflow() %>% 
  add_model(mlp_model) %>% 
  add_recipe(loan_recipe)

# tuning/fitting ----
tic("Neural Network")

mlp_tune <- mlp_workflow %>% 
  tune_grid(
    resamples = loan_fold, 
    grid = mlp_grid
    )

toc(log = TRUE)

# save runtime info
mlp_runtime <- tic.log(format = TRUE)

# write out results & workflow
save(mlp_tune, mlp_workflow, mlp_runtime, file = "model_info/mlp_tune.rda")

