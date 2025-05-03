import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

    # Initialize KNN model
    KNN_model = KNeighborsClassifier(n_neighbors=5)

    # Train the model on the entire training dataset
    KNN_model.fit(x_train, y_train)

    # Predict on test dataset
    y_pred = KNN_model.predict(x_test)
    y_pred_train = KNN_model.predict(x_train)

    # Evaluate the model
    accuracy, sensitivity, specificity = perform(KNN_model, x_test, y_test, y_pred, y_train, x_train)
    sum_accuracy += accuracy
    sum_sensitivity += sensitivity
    sum_specificity += specificity

print("sum_accuracy, sum_sensitivity, sum_specificity:", sum_accuracy/5, sum_sensitivity/5, sum_specificity/5)