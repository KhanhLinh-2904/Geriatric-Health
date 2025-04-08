from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

def perform(model, x_test, y_test, y_pred, y_train, x_train ):
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Compute sensitivity and specificity from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()  # for binary classification
    sensitivity = tp / (tp + fn)  # Recall
    specificity = tn / (tn + fp)

    # AUROC calculation
    y_test_bin = label_binarize(y_test, classes=[0, 1])  # Binarize true labels for AUROC
    y_pred_prob = model.predict_proba(x_test)[:, 1]  # Probabilities for the positive class
    auc_roc_test = roc_auc_score(y_test_bin, y_pred_prob)

    # Plot ROC Curve for test set
    fpr, tpr, _ = roc_curve(y_test_bin, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {auc_roc_test:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Test Set')
    plt.legend(loc='lower right')
    plt.show()

    # AUROC calculation for training set
    y_train_bin = label_binarize(y_train, classes=[0, 1])  # Binarize true labels for AUROC
    y_pred_prob_train = model.predict_proba(x_train)[:, 1]  # Probabilities for the positive class
    auc_roc_train = roc_auc_score(y_train_bin, y_pred_prob_train)

    # Plot ROC Curve for training set
    fpr_train, tpr_train, _ = roc_curve(y_train_bin, y_pred_prob_train)
    plt.figure()
    plt.plot(fpr_train, tpr_train, color='red', label=f'ROC curve (AUC = {auc_roc_train:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Training Set')
    plt.legend(loc='lower right')
    plt.show()

    # Display results
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Test AUROC: {auc_roc_test:.4f}")
    print(f"Training AUROC: {auc_roc_train:.4f}")
