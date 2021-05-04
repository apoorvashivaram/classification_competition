# Elastic Net tuning ----

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# seed
set.seed(20583)

# load required objects ----
load("model_info/loan_setup.rda")

# define model ----
en_model <- logistic_reg(penalty = tune(),
                         mixture = tune()) %>% 
  set_engine("glmnet")

# check tuning parameters
en_params <- parameters(en_model)

# define tuning grid ---- 
en_grid <- grid_regular(en_params, levels = 5)

# workflow ----
en_workflow <- workflow() %>% 
  add_model(en_model) %>% 
  add_recipe(loan_recipe)

# tuning/fitting ----
tic("Elastic Net")

en_tune <- en_workflow %>% 
  tune_grid(
    resamples = loan_fold, 
    grid = en_grid
    )

toc(log = TRUE)

# save runtime info
en_runtime <- tic.log(format = TRUE)

# write out results & workflow
save(en_tune, en_workflow, en_runtime, file = "model_info/en_tune.rda")

