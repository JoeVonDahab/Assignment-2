# Assignment 2: Logistic Regression

## Overview
This project implements a logistic regression model to classify patients as either non-small cell lung cancer (NSCLC) or small cell lung cancer using selected medical features.

## Structure
data/NSCLCdata.csv: Dataset file (modifiable)
regression/ containing logreg.py (logistic regression implementation) and utils.py (data loading and utilities)
test/test_logreg.py: Unit tests for logistic regression
main.py: Main script for training and evaluating the model
requirements.txt: Required packages
## Setup
Clone the Repository
Run git clone https://github.com/JoeVonDahab/Assignment-2/ and navigate to the project directory with cd Assignment-2/.

Create a Virtual Environment
With Conda, run conda create -n logistic_regression_env python=3.8 and activate it with conda activate logistic_regression_env.

Install Dependencies
Run pip install -r requirements.txt.

Ensure Data Availability
Place NSCLCdata.csv in the data/ directory.

## Running the Model
Execute main.py to train and evaluate the logistic regression model by running python main.py.

Main Script Overview
The script loads and preprocesses data, initializes the model with hyperparameters, trains the model, and evaluates performance. Key outputs include predicted probabilities, validation accuracy, and a loss history plot.

## Running Unit Tests
Run pytest in the terminal to execute the unit tests in test/test_logreg.py.

Customization
To adjust features or hyperparameters, edit main.py:

Modify the features list to include other dataset columns.
Adjust log_model initialization values like learning_rate, max_iter, and batch_size.
