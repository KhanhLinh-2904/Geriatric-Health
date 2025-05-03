import numpy as np
from sklearn.ensemble import RandomForestClassifier
from perform import perform 
import os

sum_accuracy = 0
sum_sensitivity = 0
sum_specificity = 0
dataset_fold = "splits"
files = os.listdir(dataset_fold)
for data in files:
    data_path = os.path.join(dataset_fold, data)
    best_model = os.path.join("best_model", data + "_best_model.pth")
    # Load train and test data
    loaded_data = np.load(data_path)
    x_train, x_test = loaded_data["x_train"], loaded_data["x_test"]
    y_train, y_test = loaded_data["y_train"], loaded_data["y_test"]

    # Initialize RandomForest model
    RF_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model on the entire training dataset
    RF_model.fit(x_train, y_train)

    # Predict on test dataset
    y_pred = RF_model.predict(x_test)
    y_pred_train = RF_model.predict(x_train)

    # Evaluate the model
    accuracy, sensitivity, specificity =    perform(RF_model, x_test, y_test, y_pred, y_train, x_train)
    sum_accuracy += accuracy
    sum_sensitivity += sensitivity
    sum_specificity += specificity

print("sum_accuracy, sum_sensitivity, sum_specificity:", sum_accuracy/5, sum_sensitivity/5, sum_specificity/5)

