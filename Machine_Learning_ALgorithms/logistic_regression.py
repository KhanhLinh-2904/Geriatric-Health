import numpy as np
from sklearn.linear_model import LogisticRegression
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

    # Initialize Logistic Regression model
    LR_model = LogisticRegression(max_iter=10000, random_state=42)

    # Train the model on the entire training dataset
    LR_model.fit(x_train, y_train)

    # Predict on test dataset
    y_pred = LR_model.predict(x_test)
    y_pred_train = LR_model.predict(x_train)

    # Evaluate the model
    perform(LR_model, x_test, y_test, y_pred, y_train, x_train)