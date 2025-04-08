import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from CNN1D import CNN1D

dataset_fold = "splits"
files = os.listdir(dataset_fold)
for data in files:
    data_path = os.path.join(dataset_fold, data)
    best_model = os.path.join("best_model", data + "_best_model.pth")
    # Load train and test data
    loaded_data = np.load(data_path)
    x_train, x_test = loaded_data["x_train"], loaded_data["x_test"]
    y_train, y_test = loaded_data["y_train"], loaded_data["y_test"]

    # Convert data to PyTorch tensors
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D(input_size=x_test.shape[1]).to(device)
    model.load_state_dict(torch.load(best_model, map_location=device))
    model.eval()

    # -------- Evaluate on Test Set --------
    with torch.no_grad():
        y_pred_logits_test = model(x_test_tensor.to(device))
        y_pred_test = torch.sigmoid(y_pred_logits_test).cpu().numpy()
        y_pred_labels_test = (y_pred_test > 0.5).astype(int)

    # Compute metrics on test set
    accuracy_test = accuracy_score(y_test, y_pred_labels_test)
    conf_matrix_test = confusion_matrix(y_test, y_pred_labels_test)
    tn, fp, fn, tp = conf_matrix_test.ravel()
    sensitivity_test = tp / (tp + fn)
    specificity_test = tn / (tn + fp)
    auc_test = roc_auc_score(y_test, y_pred_test)

    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test)

    # -------- Evaluate on Train Set --------
    with torch.no_grad():
        y_pred_logits_train = model(x_train_tensor.to(device))
        y_pred_train = torch.sigmoid(y_pred_logits_train).cpu().numpy()
        y_pred_labels_train = (y_pred_train > 0.5).astype(int)

    accuracy_train = accuracy_score(y_train, y_pred_labels_train)
    auc_train = roc_auc_score(y_train, y_pred_train)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train)

    # -------- Plot ROC Curves --------
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, color='red', label=f'Train ROC (AUC = {auc_train:.4f})')
    plt.plot(fpr_test, tpr_test, color='blue', label=f'Test ROC (AUC = {auc_test:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Train vs Test)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------- Print Metrics --------
    print("=== Test Set Metrics ===")
    print(f"Accuracy     : {accuracy_test:.4f}")
    print(f"Sensitivity  : {sensitivity_test:.4f}")
    print(f"Specificity  : {specificity_test:.4f}")
    print(f"AUC (Test)   : {auc_test:.4f}")

    print("\n=== Train Set Metrics ===")
    print(f"Accuracy     : {accuracy_train:.4f}")
    print(f"AUC (Train)  : {auc_train:.4f}")
