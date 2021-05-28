# classification_competition

Repo for submission to STAT 301-3 Classification Competition

# file organization

Folders `set_1` and `set_2` contain model setup files and tuning scripts for two sets of recipes for the kaggle competition. The random forest models from each of these sets were the final submissions for the kaggle competition. Each folder also contains a `kaggle_submission` folder that contains the predictions from random forest models that were submitted to the kaggle competition.

# final models

The final models for the Kaggle competition can be found under the final_models folder. To run these scripts, please follow these instructions:

-   To run each model, open the respective model setup file (`final_model_1` or `final_model_2`).

-   Run these scripts up to and including the "save necessary objects for tuning" step (approximately at line 71-72).

-   Next, move to the random forest tuning scripts for each model (`rf_tuning_1` or `rf_tuning_2` respectively).

-   Run these RF tuning scripts as individual jobs. When the jobs have successfully completed, move to the model setup file one last time.

-   Run the remaining code in the `final_model_1` or `final_model_2` scripts, starting at approximately line 73.

-   The predictions should be saved in the `kaggle_submission` folder that is inside the `final_models` folder.
