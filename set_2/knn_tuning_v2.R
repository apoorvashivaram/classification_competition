# K-Nearest Neighbors Model 2 tuning ----

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)

# Resolve common conflicts
tidymodels_prefer()

# set seed ----
set.seed(2113289)

# load required objects ----
load("model_info_v2/loan_setup_v2.rda")

# define model ----
knn_model <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

# # check tuning parameters
# update if question mark rather than + sign
# parameters(knn_model)

# set-up tuning grid ----
knn_params <- parameters(knn_model) 

# define grid ----
knn_grid <- grid_regular(knn_params, levels = 5)

# nearest neighbors workflow ----
knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(loan_recipe_v2)

# tuning/fitting ----
tic("Nearest Neighbors_model 2")

# tuning code
knn_tune <- knn_workflow %>% 
  tune_grid(
    resamples = loan_fold, 
    grid = knn_grid
  )

# calculate runtime info
toc(log = TRUE)

# save runtime info
knn_runtime <- tic.log(format = TRUE)

# write out results and workflow ---
save(knn_tune, knn_workflow, knn_runtime, file = "model_info_v2/knn_tune_v2.rda")
