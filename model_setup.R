# classification competition ----

# load libraries ----
library(tidyverse)
library(tidymodels)
library(skimr)

# set seed
set.seed(3485)

# load data ----
# training data
loan_train <- read_csv("data/train.csv") %>% 
  mutate(hi_int_prncp_pd = factor(hi_int_prncp_pd, levels = c(0, 1), labels = c("principal", "interest")))

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

# resampling via cross-validation ----
loan_fold <- vfold_cv(loan_train, v = 5, repeats = 3, strata = hi_int_prncp_pd)

# recipes -----
loan_recipe <- recipe(hi_int_prncp_pd ~ ., data = loan_train) %>% 
  step_rm(id) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_normalize(all_predictors()) %>% 
  step_interact(wlf ~ (.)^2)

# bake the recipes to verify
loan_recipe %>% 
  prep(loan_train) %>% 
  bake(new_data = NULL)

