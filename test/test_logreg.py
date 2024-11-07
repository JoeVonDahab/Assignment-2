import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from sklearn.metrics import accuracy_score

from regression.logreg import LogisticRegression
from regression.utils import loadDataset

@pytest.fixture(scope='module')
def data():
    X_train, X_val, y_train, y_val = loadDataset(split_percent=0.8)
    return X_train, X_val, y_train, y_val

@pytest.fixture
def model(data):
    X_train, _, _, _ = data
    num_features = X_train.shape[1]
    return LogisticRegression(num_feats=num_features)

def test_weight_updates(model, data):
    X_train, X_val, y_train, y_val = data
    initial_weights = model.W.copy()

    # Train the model
    model.train_model(X_train, y_train, X_val, y_val)

    # Check that weights have been updated
    assert not np.array_equal(initial_weights, model.W), "Weights were not updated during training."

    # Check that the training loss decreases
    assert model.loss_history_train[-1] < model.loss_history_train[0], "Training loss did not decrease."

    # Verify final loss is within a reasonable range
    final_loss = model.loss_history_train[-1]
    assert final_loss < 0.7, f"Final training loss is too high: {final_loss}"

def test_predictions(model, data):
    X_train, X_val, y_train, y_val = data

    # Train the model
    model.train_model(X_train, y_train, X_val, y_val)

    # Generate predictions
    y_pred_prob = model.make_prediction(X_val)

    # Ensure predictions are probabilities between 0 and 1
    assert np.all((y_pred_prob >= 0) & (y_pred_prob <= 1)), "Predicted probabilities are outside [0, 1]."

    # Convert probabilities to binary class predictions
    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred_class)
    assert accuracy > 0.6, f"Model accuracy is too low: {accuracy:.2f}"

def test_gradient_values(model, data):
    X_train, _, y_train, _ = data

    # Calculate gradient on a small batch
    sample_X, sample_y = X_train[:5], y_train[:5]
    # Pad sample_X
    sample_X = np.hstack([sample_X, np.ones((sample_X.shape[0], 1))])
    grad = model.calculate_gradient(sample_X, sample_y)

    # Check gradient norm
    grad_norm = np.linalg.norm(grad)
    assert 1e-3 < grad_norm < 1e3, f"Gradient norm out of expected range: {grad_norm}"

def test_loss_function():
    model = LogisticRegression(num_feats=1)
    y_true = np.array([0, 1])
    y_pred = np.array([0.25, 0.75])
    computed_loss = model.loss_function(y_true, y_pred)

    # Correct manual computation
    expected_loss = - (1 / 2) * (
        y_true[0] * np.log(y_pred[0] + 1e-15) + (1 - y_true[0]) * np.log(1 - y_pred[0] + 1e-15) +
        y_true[1] * np.log(y_pred[1] + 1e-15) + (1 - y_true[1]) * np.log(1 - y_pred[1] + 1e-15)
    )
    assert np.isclose(computed_loss, expected_loss), "Loss function computation is incorrect."

def test_make_prediction():
    model = LogisticRegression(num_feats=2)
    model.W = np.array([0.5, -0.25, 0.1])  # Including bias term
    X = np.array([[2, 3]])

    # Manually compute expected prediction
    z = np.dot(X, model.W[:-1]) + model.W[-1]
    z = np.clip(z, -500, 500)
    expected_pred = 1 / (1 + np.exp(-z))

    # Model prediction
    y_pred = model.make_prediction(X)
    assert np.isclose(y_pred, expected_pred), "make_prediction output is incorrect."

def test_overflow_handling():
    model = LogisticRegression(num_feats=2)
    model.W = np.array([1000, 1000, 1000])
    X = np.array([[1000, 1000]])
    y_pred = model.make_prediction(X)
    assert np.isclose(y_pred, 1.0), "Model fails to handle large input values correctly."

def test_nan_handling():
    X_train = np.array([[np.nan, 1], [2, 3]])
    y_train = np.array([0, 1])
    X_val = np.array([[4, 5]])
    y_val = np.array([1])
    model = LogisticRegression(num_feats=2)

    with pytest.raises(ValueError):
        model.train_model(X_train, y_train, X_val, y_val)
