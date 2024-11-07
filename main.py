import numpy as np
import pandas as pd
from regression import logreg, utils
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report

def main():
    """
    This script trains a logistic regression model on the NSCLC dataset
    to predict whether a patient has non-small cell lung cancer (NSCLC)
    or small cell lung cancer based on selected features.

    Steps:
    1. Load the dataset with specified features.
    2. Scale the features using StandardScaler.
    3. Optionally, add polynomial features to increase model capacity.
    4. Initialize the LogisticRegression model with hyperparameters.
    5. Train the model using the training data.
    6. Plot the loss history to visualize training progress.
    7. Make predictions on the validation set.
    8. Evaluate the model's performance using accuracy and classification report.
    """

    # Specify features to use
    features = [
        'Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)',
        'Low Density Lipoprotein Cholesterol',
        'Creatinine',
        'AGE_DIAGNOSIS'
    ]

    # Load dataset with specified features
    X_train, X_val, y_train, y_val = utils.loadDataset(
        features=features,
        split_percent=0.8,
        split_state=42
    )

    # Ensure labels are binary and of integer type
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)
    assert set(np.unique(y_train)).issubset({0, 1}), "y_train contains values other than 0 and 1."

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Optionally add polynomial features to increase model capacity
    # Here we use degree=2 to capture interactions between features
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)

    num_features = X_train_poly.shape[1]

    # Initialize the logistic regression model with hyperparameters
    log_model = logreg.LogisticRegression(
        num_feats=num_features,
        learning_rate=0.001,          # Reduced learning rate
        max_iter=500,                 # Increased number of epochs
        batch_size=16,
        regularization_strength=0.01  # L2 regularization strength
    )

    # Train the model
    log_model.train_model(X_train_poly, y_train, X_val_poly, y_val)

    # Plot loss history
    log_model.plot_loss_history()

    # Make predictions on validation set
    y_pred = log_model.make_prediction(X_val_poly)
    print("Predicted probabilities:", y_pred)

    # Convert probabilities to binary predictions
    y_pred_class = (y_pred >= 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred_class)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Additional performance metrics
    print("Classification Report:")
    print(classification_report(y_val, y_pred_class))

if __name__ == "__main__":
    main()
