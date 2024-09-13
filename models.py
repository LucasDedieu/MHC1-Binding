import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, input_size: int = 1024, num_classes: int = 2, dropout_prob: float = 0.3) -> None:
        """
        Initialize the ClassificationHead module.

        Args:
            input_size (int): The number of input features to the first fully connected layer. Default is 1024.
            num_classes (int): The number of output classes for classification. Default is 2.
            dropout_prob (float): The dropout probability. Default is 0.3.
        """
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x