# src/models/mlp_model.py
import torch.nn as nn
from pytorch_lightning import LightningModule

class MLPModel(LightningModule):
    def __init__(self, hidden_dim, num_layers, dropout_prob, learning_rate, window_size=None):
        super().__init__()
        self.save_hyperparameters()
        layers = []
        input_dim = window_size or 60
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)