"""Perceptron de camada única, implementado do zero (como em Perceptron.py da monografia).

Fiel ao original: ativação degrau (>= 0 -> 1), pesos e viés iniciais 0, taxa 0,01, 100 épocas, uma passagem por época em
ordem aleatória, regra w <- w + eta (y - y_hat) x. Sem pesos de classe.
Acrescenta ``decision_function`` (w·x + b, score contínuo) para ROC/PR sem fabricar curvas a partir de saídas binárias.
"""
from __future__ import annotations

import numpy as np

PERCEPTRON_PARAMS = {"learning_rate": 0.01, "n_epochs": 100, "random_state": 42}


class PerceptronScratch:
    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 100, random_state: int = 42):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.weights: np.ndarray | None = None
        self.bias: float | None = None
        self.errors_per_epoch: list[int] = []

    @staticmethod
    def activation(x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PerceptronScratch":
        rng = np.random.default_rng(self.random_state)   # gerador próprio: não altera o estado global do NumPy
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.errors_per_epoch = []
        for _ in range(self.n_epochs):
            idx = rng.permutation(n_samples)
            errors = 0
            for i in idx:
                y_hat = 1 if (X[i] @ self.weights + self.bias) >= 0 else 0
                update = self.learning_rate * (y[i] - y_hat)
                if update != 0.0:
                    self.weights += update * X[i]
                    self.bias += update
                    errors += 1
            self.errors_per_epoch.append(errors)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.activation(self.decision_function(X))
