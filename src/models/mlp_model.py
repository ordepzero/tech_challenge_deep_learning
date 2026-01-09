# src/models/mlp_model.py
import torch.nn as nn
from lightning import LightningModule

class MLPModel(LightningModule):
    """
    Modelo de Perceptron Multicamadas (Multi-Layer Perceptron) para previsão de séries temporais.
    """
    def __init__(self, hidden_dim, num_layers, dropout_prob, learning_rate, window_size=None):
        """
        Inicializa o MLP com o número especificado de camadas e dimensões ocultas.
        """
        super().__init__()
        self.save_hyperparameters()
        layers = []
        # Define a dimensão de entrada baseada na janela (padrão 60)
        input_dim = window_size or 60
        
        # Constrói as camadas ocultas dinamicamente
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            input_dim = hidden_dim
            
        # Camada de saída única para regressão (valor do próximo passo)
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Executa a passagem para frente (forward pass) dos dados pela rede MLP.
        """
        return self.net(x)
