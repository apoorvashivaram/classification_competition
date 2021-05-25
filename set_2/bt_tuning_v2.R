# boosted tree model 2 tuning ----

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# set seed ----
set.seed(56789)

# load required objects ----
load("model_info_v2/loan_setup_v2.rda")

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
  update(mtry = mtry(range = c(1, 5)),
         learn_rate = learn_rate(range = c(-5, -0.1))
         )

# define grid ----
bt_grid <- grid_regular(bt_params, levels = 5)

# boosted tree workflow ----
bt_workflow <- workflow() %>% 
  add_model(bt_model) %>% 
  add_recipe(loan_recipe_v2)

# tuning/fitting ----
tic("Boosted Tree_model 2")

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
save(bt_tune, bt_workflow, bt_runtime, file = "model_info_v2/bt_tune_v2.rda")

