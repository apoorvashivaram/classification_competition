# classification competition ----
# final model 2 ----

# load libraries ----
library(tidyverse)
library(tidymodels)
library(skimr)
library(corrplot)

# Resolve common conflicts
tidymodels_prefer()

# set seed
set.seed(12357)

# load data ----
# training data
loan_train <- read_csv("data/train.csv") %>% 
  mutate(hi_int_prncp_pd = factor(hi_int_prncp_pd, levels = c(0, 1))) %>%
  mutate_at(c("addr_state", "application_type", "emp_length", "grade",
              "home_ownership", "initial_list_status", "purpose", "sub_grade", "term", "verification_status"), as.factor)


# testing data
loan_test <- read_csv("data/test.csv")

# short EDA ----
skim_without_charts(loan_train)

# explore target variable
# high class imbalance
loan_train %>% 
  ggplot(aes(hi_int_prncp_pd)) +
  geom_bar() +
  theme_minimal()

# proportions - 20:80
loan_train %>% 
  count(hi_int_prncp_pd) %>% 
  mutate(prop = n / sum(n))

# check for missingness - no missing data
naniar::any_miss(loan_train)
naniar::any_miss(loan_test)

# corrplot
loan_train %>% 
  select(-c(addr_state, earliest_cr_line, emp_title, last_credit_pull_d)) %>% 
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.integer) %>%
  cor() %>% 
  corrplot(method = "circle")

# resampling via cross-validation ----
loan_fold <- vfold_cv(loan_train, v = 5, repeats = 3, strata = hi_int_prncp_pd)

# recipes -----
loan_recipe_v2 <- recipe(hi_int_prncp_pd ~ initial_list_status + int_rate + loan_amnt + 
                           out_prncp_inv + term,
                         data = loan_train) %>% 
  step_other(all_nominal(), -all_outcomes(), threshold = 0.1) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_normalize(all_predictors()) %>% 
  step_zv(all_predictors())

# bake the recipes to verify
loan_recipe_v2 %>% 
  prep(loan_train) %>% 
  bake(new_data = NULL)

# save necessary objects for tuning ----
save(loan_fold, loan_recipe_v2, file = "final_models/model_info/loan_setup_2.rda")

# tuning models & separate R scripts ----
# rf_tuning_2 script to be opened & run as a job
# random forest -- rf_tuning_2.R

# remaining models attempted, but did not use as part of the final submission
# nearest neighbors -- knn_tuning_v2.R
# boosted tree -- bt_tuning_v2.R
# single Layer Neural Network (multilayer perceptron --- mlp) -- mlp_tuning_v2.R

# load tuned files ----


# random forest
load(file = "final_models/model_info/rf_tune_2.rda")

# # nearest neighbors
# load(file = "final_models/model_info/knn_tune_v2.rda")
# 
# # boosted tree
# load(file = "final_models/model_info/bt_tune_v2.rda")
# 
# # single layer neural network - mlp
# load(file = "final_models/model_info/mlp_tune_v2.rda")


# autoplots ----

rf_tune %>% 
  autoplot(metric = "accuracy")

# knn_tune %>% 
#   autoplot(metric = "accuracy")
# 
# bt_tune %>% 
#   autoplot()
# 
# mlp_tune %>% 
#   autoplot(metric = "accuracy")

# select best model
tune_results <- tibble(
  model_type = c(# "Nearest Neighbors", 
                 "Random Forest" #, 
                 # "Boosted Tree",
                 # "Neural Network"
  ),
  tune_info = list(# knn_tune, 
                   rf_tune # , 
                   # bt_tune, 
                   # mlp_tune
  ),
  assessment_info = map(tune_info, collect_metrics),
  best_model = map(tune_info, ~ select_best(.x, metric = "accuracy"))
)

# select best models
tune_results %>% 
  select(model_type, best_model) %>% 
  unnest(best_model)


# create a table with runtimes
tune_runtime <- tibble(
  run_time = list(
    knn_runtime, 
    rf_runtime, 
    bt_runtime,
    mlp_runtime
  ),
) %>% 
  unnest_wider(run_time) %>% 
  rename(run_time = "...1") %>% 
  separate(run_time, into = c("model_type", "run_time"), sep = ": ") %>% 
  separate(run_time, into = c("run_time", "sec", "elapsed"), sep = " ") %>% 
  select(-c("sec", "elapsed"))

# combine best model results and run_time table together
tune_results %>% 
  select(model_type, assessment_info) %>% 
  unnest(assessment_info) %>% 
  filter(.metric == "roc_auc") %>% 
  arrange(desc(mean)) %>% 
  left_join(tune_runtime, by = c("model_type")) %>% 
  group_by(model_type) %>% 
  arrange(desc(mean)) %>% 
  distinct(model_type, .keep_all = T) %>% 
  select(model_type, run_time, .metric, mean, std_err) %>% 
  mutate(mean = round(mean, 4),
         std_err = round(std_err, 4)) %>% 
  rename("model type" = model_type, "run time (sec)" = run_time, metric = .metric, "std error" = std_err) %>% 
  DT::datatable()

# finalize rf_workflow
rf_workflow_tuned <- rf_workflow %>% 
  finalize_workflow(select_best(rf_tune, metric = "accuracy"))

# fit to training data
rf_results <- fit(rf_workflow_tuned, loan_train)

# predict on testing data
final_rf_results <- rf_results %>%
  predict(new_data = loan_test) %>%
  bind_cols(loan_test %>%
              select(id)) %>%
  mutate(Category = .pred_class,
         Id = id) %>%
  select(Id, Category)

# check final results
final_rf_results

# # write out file for kaggle submission
# write_csv(final_rf_results, "final_models/kaggle_submission/rf_output_2.csv")
