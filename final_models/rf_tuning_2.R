# Random Forest tuning ----
# for MODEL 2

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# set seed ----
set.seed(782235)

# load required objects ----
load("model_info/loan_setup_2.rda")

# define model ----
rf_model <- rand_forest(
  mode = "classification", 
  mtry = tune(), 
  min_n = tune()
) %>% 
  # variable importance plot
  set_engine("ranger", importance = "impurity")

# # check tuning parameters
# parameters(rf_model)

# set-up tuning grid ----
rf_params <- parameters(rf_model) %>% 
  # don't want to use all the parameters (# of predictors)
  update(mtry = mtry(range = c(1, 5)))

# define grid ----
rf_grid <- grid_regular(rf_params, levels = 5)

# random forest workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(loan_recipe_v2)

# tuning/fitting ----
tic("Random Forest_model 2")

# tuning code
rf_tune <- rf_workflow %>% 
  tune_grid(
    resamples = loan_fold, 
    grid = rf_grid
  )

# calculate runtime info
toc(log = TRUE)

# save runtime info
rf_runtime <- tic.log(format = TRUE)

# write out results and workflow ---
save(rf_tune, rf_workflow, rf_runtime, file = "final_models/model_info/rf_tune_2.rda")