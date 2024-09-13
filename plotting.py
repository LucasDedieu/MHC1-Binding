import matplotlib.pyplot as plt
import pandas as pd
import umap
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc


def train_test_histogram(df_train: pd.DataFrame, df_test: pd.DataFrame, column_name: str, figsize: 'tuple[int, int]') -> None:
    """
    Plots a histogram comparing the frequency of values in a specified column
    between the training and testing datasets.

    Args:
        df_train (pd.DataFrame): Training dataset.
        df_test (pd.DataFrame): Testing dataset.
        column_name (str): The column name to compare.
        figsize (tuple[int, int]): Figure size for the plot.
    """
    train_counts = df_train[column_name].value_counts().sort_index()
    test_counts = df_test[column_name].value_counts().sort_index()
    combined_counts = pd.DataFrame({'train': train_counts, 'test': test_counts}).fillna(0)

    plt.figure(figsize=figsize)

    combined_counts['train'].plot(kind='bar', color='purple', label='train')
    combined_counts['test'].plot(kind='bar', color='orange', label='test')

    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.title('Histogram for column ' + column_name)
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()


def plot2d(embeddings: np.ndarray, labels: np.ndarray, embd_method: str, method: str = 'tsne', seed: int = 42) -> None:
    """
    Plots 2D embeddings using t-SNE or UMAP for dimensionality reduction.

    Args:
        embeddings (np.ndarray): High-dimensional embeddings to be reduced.
        labels (np.ndarray): Labels corresponding to the embeddings.
        embd_method (str): Description of the embedding method used.
        method (str, optional): The dimensionality reduction method ('tsne' or 'umap'). Defaults to 'tsne'.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=seed)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=seed)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    embeddings_2d = reducer.fit_transform(embeddings)

    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', s=1)
    plt.title(f"{method.upper()} of {embd_method}")
    legend = plt.legend(*scatter.legend_elements(), title="Labels")
    plt.gca().add_artist(legend)
    plt.show()


def plot2d_train_test(embeddings_train: np.ndarray, embeddings_test: np.ndarray, embd_method: str, method: str = 'tsne', seed: int = 42) -> None:
    """
    Plots 2D embeddings for both training and testing datasets using t-SNE or UMAP for dimensionality reduction.

    Args:
        embeddings_train (np.ndarray): High-dimensional embeddings for the training dataset.
        embeddings_test (np.ndarray): High-dimensional embeddings for the testing dataset.
        embd_method (str): Description of the embedding method used.
        method (str, optional): The dimensionality reduction method ('tsne' or 'umap'). Defaults to 'tsne'.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=seed)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=seed)
    else:
        raise ValueError("Method must be 'tsne' or 'umap'")

    combined_embeddings = np.vstack((embeddings_train, embeddings_test))

    embeddings_2d = reducer.fit_transform(combined_embeddings)
    embeddings_2d_train = embeddings_2d[:len(embeddings_train)]
    embeddings_2d_test = embeddings_2d[len(embeddings_train):]

    plt.scatter(embeddings_2d_train[:, 0], embeddings_2d_train[:, 1], c='purple', s=1, label='Train')
    plt.scatter(embeddings_2d_test[:, 0], embeddings_2d_test[:, 1], c='orange', s=1, label='Test')

    plt.title(f"{method.upper()} of {embd_method}")
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plots confusion matrix between labels and predicted labels.

    Args:
        y_test (np.ndarray): Labels of test dataset.
        y_pred (np.ndarray): Predicted lalels for test dataset.
    """
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def plot_precision_recall_curve(y_test: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Plot the Precision-Recall curve and calculate the area under the curve (AUC).

    Args:
        y_test (np.ndarray): Array of true binary labels (0 or 1).
        y_prob (np.ndarray): Array of predicted probabilities for the positive class.

    Returns:
        float: The area under the Precision-Recall curve (AUC).
    """
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(4, 3))
    plt.plot(recall, precision, label='AUC = {:.2f}'.format(pr_auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    
    return pr_auc
