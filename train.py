import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from typing import Callable, Tuple, List, Optional


def train_model(
    device: torch.device,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: optim.Optimizer,
    scheduler: Optional[_LRScheduler],
    n_epoch: int,
    patience: int = 10
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """
    Train a given model using the specified parameters and data loaders.

    Args:
        device (torch.device): Device to run the training on (e.g., CPU or GPU).
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        criterion (callable): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        n_epoch (int): Number of epochs to train the model.
        patience (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 10.

    Returns:
        tuple: A tuple containing lists of training loss, validation loss, validation accuracy, precision, recall, and F1 score.
    """
    t_loss = []
    v_loss = []
    v_acc = []
    v_f1 = []
    v_precision = []
    v_recall = []
    epochs_before_stop = patience
    loss_func = criterion
    best_v_loss = float('inf')

    for epoch in range(n_epoch):
        running_loss = 0.0
        val_loss = 0.0
        correct = 0
        total = 0

        # Train
        model.train()
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch}"):
            images, labels = data
            labels = labels.type(torch.LongTensor)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        # Validation
        predictions = []
        true_labels = []
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                labels = labels.type(torch.LongTensor)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = loss_func(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = balanced_accuracy_score(true_labels, predictions)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, pos_label=1, average="binary", zero_division=0)
        v_precision.append(precision)
        v_recall.append(recall)
        v_f1.append(f1)
        v_acc.append(accuracy)
        t_loss.append(epoch_train_loss)
        v_loss.append(epoch_val_loss)
        print(f'epoch {epoch + 1} -> t_loss: {epoch_train_loss:.4f}, v_loss: {epoch_val_loss:.3f}, bacc: {accuracy:.4f}, precision:{precision:.2f}, recall:{recall:.2f}, f1:{f1:.2f}')

        if scheduler:
            scheduler.step()

        # Early stopping
        if epoch_val_loss < best_v_loss:
            best_v_loss = epoch_val_loss
            epochs_before_stop = patience
        else:
            epochs_before_stop -= 1

        if epochs_before_stop == 0:
            print('Early Stopping:', patience, 'epochs since last best val loss. Total epochs trained:', epoch + 1)
            return t_loss, v_loss, v_acc, v_precision, v_recall, v_f1
        
    return t_loss, v_loss, v_acc, v_precision, v_recall, v_f1
