# logreg.py

import sys
import os

# Add the directory containing this script to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import loadDataset  # Import remains unchanged

def sigmoid(z):
    """
    Numerically stable sigmoid function.
    """
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    result = np.zeros_like(z)
    
    # For positive z, no risk of overflow
    result[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
    
    # For negative z, to avoid overflow in exp(z)
    exp_z = np.exp(z[neg_mask])
    result[neg_mask] = exp_z / (1 + exp_z)
    
    return result

class BaseRegressor:
    def __init__(self, num_feats, learning_rate=0.001, tol=0.0001, max_iter=500, batch_size=16):
        self.W = np.random.randn(num_feats + 1).flatten() * 0.01  # Including bias term
        self.lr = learning_rate
        self.tol = tol
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_feats = num_feats  # Number of features without bias term
        self.loss_history_train = []
        self.loss_history_val = []

    def calculate_gradient(self, X, y):
        raise NotImplementedError("Subclass must implement this method.")

    def loss_function(self, y_true, y_pred):
        raise NotImplementedError("Subclass must implement this method.")

    def make_prediction(self, X):
        raise NotImplementedError("Subclass must implement this method.")

    def train_model(self, X_train, y_train, X_val, y_val):
        # Check for NaN values in the training data
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            raise ValueError("Training data contains NaN values.")

        # Padding data with a column of ones for the bias term
        X_train = self._add_bias(X_train)
        X_val = self._add_bias(X_val)

        prev_update_size = 1
        iteration = 1

        while prev_update_size > self.tol and iteration <= self.max_iter:
            # Shuffle the training data
            shuffle_indices = np.random.permutation(len(y_train))
            X_train_shuffled = X_train[shuffle_indices]
            y_train_shuffled = y_train[shuffle_indices]

            num_batches = int(np.ceil(X_train.shape[0] / self.batch_size))
            X_batches = np.array_split(X_train_shuffled, num_batches)
            y_batches = np.array_split(y_train_shuffled, num_batches)

            update_size_epoch = []
            epoch_loss_train = []

            for X_batch, y_batch in zip(X_batches, y_batches):
                y_pred = self.make_prediction(X_batch)
                loss_train = self.loss_function(y_batch, y_pred)
                epoch_loss_train.append(loss_train)

                prev_W = self.W.copy()
                grad = self.calculate_gradient(X_batch, y_batch)
                self.W -= self.lr * grad  # Update weights

                update_size_epoch.append(np.linalg.norm(self.W - prev_W))

            # Calculate average training loss for the epoch
            avg_loss_train = np.mean(epoch_loss_train)
            self.loss_history_train.append(avg_loss_train)

            # Calculate validation loss
            y_val_pred = self.make_prediction(X_val)
            loss_val = self.loss_function(y_val, y_val_pred)
            self.loss_history_val.append(loss_val)

            prev_update_size = np.mean(update_size_epoch)
            iteration += 1

            # Optional: print losses for monitoring
            print(f"Epoch {iteration-1}, Training Loss: {avg_loss_train:.4f}, Validation Loss: {loss_val:.4f}")

        print("Training completed.")

    def plot_loss_history(self):
        assert len(self.loss_history_train) > 0, "Need to run training before plotting loss history"

        fig, axs = plt.subplots(2, figsize=(8, 8))
        fig.suptitle('Loss History')
        axs[0].plot(self.loss_history_train)
        axs[0].set_title('Training Loss')
        axs[1].plot(self.loss_history_val)
        axs[1].set_title('Validation Loss')
        axs[1].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[1].set_ylabel('Loss')
        plt.tight_layout()
        plt.show()

    def _add_bias(self, X):
        """
        Add a column of ones to the input data matrix to account for the bias term.
        """
        return np.hstack([X, np.ones((X.shape[0], 1))])

class LogisticRegression(BaseRegressor):
    def __init__(self, num_feats, learning_rate=0.001, tol=0.0001, max_iter=500, batch_size=16, regularization_strength=0.01):
        super().__init__(num_feats, learning_rate, tol, max_iter, batch_size)
        self.regularization_strength = regularization_strength  # L2 regularization strength

    def calculate_gradient(self, X, y) -> np.ndarray:
        m = X.shape[0]
        z = np.dot(X, self.W)
        prediction = sigmoid(z)
        error = prediction - y
        # Include L2 regularization term
        gradient = (1 / m) * np.dot(X.T, error) + self.regularization_strength * self.W
        return gradient

    def loss_function(self, y_true, y_pred) -> float:
        m = y_true.shape[0]
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        # Include L2 regularization term
        regularization_term = (self.regularization_strength / 2) * np.sum(self.W ** 2)
        loss = - (1 / m) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        ) + regularization_term
        return loss

    def make_prediction(self, X) -> np.ndarray:
        # Pad X with bias term if not already padded
        if X.shape[1] == self.num_feats:
            X = self._add_bias(X)
        z = np.dot(X, self.W)
        y_pred = sigmoid(z)
        return y_pred

if __name__ == "__main__":
    # Load dataset
    from sklearn.preprocessing import StandardScaler

    # Specify features to use
    features = [
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)',
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'
    ]

    X_train, X_val, y_train, y_val = loadDataset(features=features, split_percent=0.8)

    # Ensure labels are binary and of integer type
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    assert set(np.unique(y_train)).issubset({0, 1}), "y_train contains values other than 0 and 1."

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Optionally add polynomial features to increase model capacity
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    num_features = X_train_poly.shape[1]
    log_reg_model = LogisticRegression(
        num_feats=num_features,
        learning_rate=0.001,          # Reduced learning rate
        max_iter=500,                 # Increased number of epochs
        batch_size=16,
        regularization_strength=0.01  # Added regularization strength
    )

    log_reg_model.train_model(X_train_poly, y_train, X_val_poly, y_val)
    log_reg_model.plot_loss_history()

    # Making predictions on validation set
    y_pred = log_reg_model.make_prediction(X_val_poly)
    print("Predicted probabilities:", y_pred)

    # Convert probabilities to binary predictions
    y_pred_class = (y_pred >= 0.5).astype(int)

    # Calculate accuracy
    from sklearn.metrics import accuracy_score, classification_report
    accuracy = accuracy_score(y_val, y_pred_class)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Additional performance metrics
    print("Classification Report:")
    print(classification_report(y_val, y_pred_class))


