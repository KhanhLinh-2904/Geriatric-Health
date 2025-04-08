import numpy as np
import xgboost as xgb
from perform import perform 
import os

dataset_fold = "splits"
files = os.listdir(dataset_fold)
for data in files:
    data_path = os.path.join(dataset_fold, data)
    best_model = os.path.join("best_model", data + "_best_model.pth")
    # Load train and test data
    loaded_data = np.load(data_path)
    x_train, x_test = loaded_data["x_train"], loaded_data["x_test"]
    y_train, y_test = loaded_data["y_train"], loaded_data["y_test"]

    # Initialize XGBoost model
    XGB_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    # Train the model on the entire training dataset
    XGB_model.fit(x_train, y_train)

    # Predict on test dataset
    y_pred = XGB_model.predict(x_test)
    y_pred_train = XGB_model.predict(x_train)

    # Evaluate the model
    perform(XGB_model, x_test, y_test, y_pred, y_train, x_train)