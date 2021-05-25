# Multivariate adaptive regression splines (MARS) tuning ----

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# seed
set.seed(545)

# load required objects ----
load("model_info/loan_setup.rda")

# define model ----
mars_model <-
  # specify model type and parameters to optimize
  mars(num_terms = tune(),
       prod_degree = tune()) %>% 
  # set mode
  set_mode("classification") %>% 
  # set underlying engine/package
  set_engine("earth")

# check tuning parameters
mars_params <- parameters(mars_model) %>% 
  update(num_terms = num_terms(range = c(1, 10)))

# define tuning grid ---- 
mars_grid <- grid_regular(mars_params, levels = 5)

# workflow ----
mars_workflow <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(loan_recipe)

# tuning/fitting ----
tic("MARS")

mars_tune <- mars_workflow %>% 
  tune_grid(
    resamples = loan_fold, 
    grid = mars_grid
    )

toc(log = TRUE)

# save runtime info
mars_runtime <- tic.log(format = TRUE)

# write out results & workflow
save(mars_tune, mars_workflow, mars_runtime, file = "model_info/mars_tune.rda")

