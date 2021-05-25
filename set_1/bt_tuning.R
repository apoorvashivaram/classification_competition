# boosted tree tuning ----

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# set seed ----
set.seed(3729)

# load required objects ----
load("model_info/loan_setup.rda")

# define model ----
bt_model <- boost_tree(
  mode = "classification", 
  mtry = tune(), 
  min_n = tune(),
  learn_rate = tune(),
  ) %>% 
  # variable importance plot
  set_engine("xgboost", importance = "impurity")

# # check tuning parameters
# parameters(bt_model)

# set-up tuning grid ----
bt_params <- parameters(bt_model) %>% 
  # don't want to use all the parameters (# of predictors)
  update(mtry = mtry(range = c(2, 10)),
         learn_rate = learn_rate(range = c(-5, -0.2))
         )

# define grid ----
bt_grid <- grid_regular(bt_params, levels = 5)

# boosted tree workflow ----
bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(loan_recipe)

# tuning/fitting ----
tic("Boosted Tree")

# tuning code
bt_tune <- bt_workflow %>% 
  tune_grid(
    resamples = loan_fold, 
    grid = bt_grid
  )

# calculate runtime info
toc(log = TRUE)

# save runtime info
bt_runtime <- tic.log(format = TRUE)

# write out results and workflow ---
save(bt_tune, bt_workflow, bt_runtime, file = "model_info/bt_tune.rda")

