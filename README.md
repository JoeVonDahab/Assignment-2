Assignment 2: Logistic Regression
Project Overview
This project implements a logistic regression model from scratch to predict whether a patient has non-small cell lung cancer (NSCLC) or small cell lung cancer based on medical features. The model is trained and evaluated on a simulated medical record dataset.

Key Components:
Custom Logistic Regression Implementation: Includes gradient calculation, loss function, and prediction methods.
Data Preprocessing: Feature scaling and polynomial feature expansion to enhance model capacity.
Model Training and Evaluation: Training loop with L2 regularization and performance metrics.
Unit Tests: Comprehensive tests to ensure the correctness of the implementation.
Table of Contents
Project Structure
Prerequisites
Installation
Running the Model
Understanding the Output
Running Unit Tests
Customizing the Model
Dataset Information
References
Project Structure
bash
Copy code
project_directory/
├── data/
│   └── NSCLCdata.csv
├── regression/
│   ├── __init__.py
│   ├── logreg.py
│   └── utils.py
├── test/
│   └── test_logreg.py
├── main.py
├── requirements.txt
└── README.md
data/NSCLCdata.csv: The dataset containing patient medical records.
regression/: Package containing the logistic regression implementation and utilities.
logreg.py: Implementation of the logistic regression model.
utils.py: Utility functions, including dataset loading.
test/test_logreg.py: Unit tests for the logistic regression model.
main.py: Script to train and evaluate the logistic regression model.
requirements.txt: List of required Python packages.
README.md: Instructions and information about the project.
Prerequisites
Python 3.6 or higher
Anaconda or Miniconda (Recommended): For managing virtual environments.
Git: For cloning the repository.
Installation
1. Clone the Repository
Open a terminal and clone the repository:

bash
Copy code
git clone https://github.com/your_username/your_repository.git
cd your_repository
2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.

Using Conda:
bash
Copy code
conda create -n logistic_regression_env python=3.8
conda activate logistic_regression_env
Using venv:
bash
Copy code
python -m venv venv
# Activate the environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
3. Install Dependencies
Install the required packages using pip:

bash
Copy code
pip install -r requirements.txt
If you don't have a requirements.txt file, the essential packages are:

bash
Copy code
pip install numpy pandas scikit-learn matplotlib pytest
4. Ensure Data Availability
Ensure that the NSCLCdata.csv file is located in the data/ directory:

kotlin
Copy code
data/
└── NSCLCdata.csv
Running the Model
To train and evaluate the logistic regression model, run the main.py script:

bash
Copy code
python main.py
What Happens When You Run main.py
The script performs the following steps:

Data Loading: Loads the NSCLC dataset with specified features.
Data Preprocessing:
Scales the features using StandardScaler.
Adds polynomial features to capture non-linear relationships.
Model Initialization: Initializes the logistic regression model with hyperparameters.
Model Training: Trains the model on the training data, printing loss at each epoch.
Model Evaluation:
Makes predictions on the validation set.
Calculates validation accuracy and displays a classification report.
Visualization: Plots the training and validation loss over epochs.
Understanding the Output
Training Progress
During training, the script prints the training and validation loss at each epoch:

yaml
Copy code
Epoch 1, Training Loss: 0.6931, Validation Loss: 0.6928
Epoch 2, Training Loss: 0.6915, Validation Loss: 0.6911
...
Epoch 500, Training Loss: 0.2994, Validation Loss: 0.2907
Training completed.
Predicted Probabilities
After training, the script outputs the predicted probabilities for the validation set:

less
Copy code
Predicted probabilities: [0.99977604 0.21581652 0.98388893 ...]
Validation Accuracy
The script calculates and prints the validation accuracy:

mathematica
Copy code
Validation Accuracy: 0.93
Classification Report
A detailed performance report is displayed:

markdown
Copy code
Classification Report:
              precision    recall  f1-score   support

           0       0.88      0.99      0.93       199
           1       0.99      0.86      0.92       201

    accuracy                           0.93       400
   macro avg       0.93      0.93      0.92       400
weighted avg       0.93      0.93      0.92       400
Loss History Plot
A plot window will appear showing the training and validation loss over epochs:


Running Unit Tests
Unit tests are provided to ensure the correctness of the logistic regression implementation.

Run Tests with Pytest
Ensure you have pytest installed:

bash
Copy code
pip install pytest
Run the tests:

bash
Copy code
pytest
Expected Output
bash
Copy code
================================================= test session starts =================================================
platform win32 -- Python 3.x.x, pytest-7.4.4, pluggy-1.0.0
rootdir: path_to_project_directory
collected 7 items

test\test_logreg.py .......                                                                                     [100%]

================================================== 7 passed in 5.63s ==================================================
Customizing the Model
Changing Features
Modify the features list in main.py to include different features from the dataset:

python
Copy code
features = [
    'New Feature 1',
    'New Feature 2',
    # Add or remove features as needed
]
Adjusting Hyperparameters
Experiment with different hyperparameters in the LogisticRegression initialization:

python
Copy code
log_model = logreg.LogisticRegression(
    num_feats=num_features,
    learning_rate=0.001,          # Adjust learning rate
    max_iter=500,                 # Adjust number of epochs
    batch_size=16,                # Adjust batch size
    regularization_strength=0.01  # Adjust regularization strength
)
Polynomial Features
Change the degree of polynomial features to capture higher-order interactions:

python
Copy code
poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
Dataset Information
The dataset contains simulated medical records with various features:

Target Variable: NSCLC (1 = Non-Small Cell Lung Cancer, 0 = Small Cell Lung Cancer)
Available Features (not exhaustive):
GENDER
Penicillin V Potassium 250 MG
Penicillin V Potassium 500 MG
Computed tomography of chest and abdomen
Plain chest X-ray (procedure)
Diastolic Blood Pressure
Body Mass Index
Body Weight
Body Height
Systolic Blood Pressure
Low Density Lipoprotein Cholesterol
High Density Lipoprotein Cholesterol
Triglycerides
Total Cholesterol
AGE_DIAGNOSIS
... and more.
You can view all features by inspecting the dataset or the loadDataset function in utils.py.

References
Logistic Regression Theory:
Understanding Logistic Regression
Binary Cross-Entropy Loss:
Understanding Binary Cross-Entropy Loss
Sigmoid Function:
Derivative of the Sigmoid Function
Dataset Reference:
Synthetic Data Generation for Medical Datasets
Troubleshooting
Import Errors: Ensure that the regression package contains an __init__.py file.
Data File Not Found: Verify that NSCLCdata.csv is in the data/ directory.
Module Not Found: Adjust the PYTHONPATH or use sys.path.append() to include the project directory.
Plot Not Displaying: Ensure matplotlib is installed and configured correctly.