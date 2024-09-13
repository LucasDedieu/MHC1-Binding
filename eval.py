import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
import plotting
import torch
from typing import List


def eval_model(device: torch.device, model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> None:
    """
    Evaluate the model using the given data loader and compute metrics.

    Args:
        device (torch.device): The device to run the evaluation on (e.g., CPU or GPU).
        model (torch.nn.Module): The model to be evaluated.
        loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
    """
    predictions = []
    true_labels = []
    probas = []

    model.eval()
    with torch.no_grad():
        for data in loader:
            features, labels = data
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            probas.extend(outputs.cpu().numpy())
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    y_pred = np.array(predictions)
    y_test = np.array(true_labels)
    compute_metrics(y_test, y_pred, probas)


def compute_metrics(y_test: np.ndarray, y_pred: np.ndarray, y_prob: List[np.ndarray]) -> None:
    """
    Compute and print various classification metrics including accuracy, F1 score,
    classification report, and Precision-Recall AUC.

    Args:
        y_test (np.ndarray): Array of true binary labels.
        y_pred (np.ndarray): Array of predicted labels.
        y_prob (List[np.ndarray]): List of predicted probabilities for the positive class.
    """
    accuracy = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    print("Classification Report:\n", report)
    print("Balanced Accuracy:", accuracy)
    print("F1 Score :", f1)

    pr_auc = plotting.plot_precision_recall_curve(y_test, np.array(y_prob)[:,1])
    print("PR AUC :", pr_auc)

    plotting.plot_confusion_matrix(y_test, y_pred)