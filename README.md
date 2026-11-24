# Breast Cancer Prediction Using Gradient Boosting

A machine learning project that classifies breast cancer observations as
**benign** or **malignant** using a Gradient Boosting Classifier. The
complete implementation and visualizations are provided in
`cancer_prediction.ipynb`.

> **Disclaimer:** This is an academic machine learning project. It is
> not a medical diagnostic tool and must not be used to make healthcare
> decisions.

## Table of Contents

-   [About the Project](#about-the-project)
-   [Dataset](#dataset)
-   [Technologies Used](#technologies-used)
-   [Project Workflow](#project-workflow)
-   [Machine Learning Model](#machine-learning-model)
-   [Model Evaluation](#model-evaluation)
-   [Project Structure](#project-structure)
-   [Installation](#installation)
-   [How to Run](#how-to-run)
-   [Example Prediction](#example-prediction)

## About the Project

The objective is to build and evaluate a supervised machine learning
classification model using numerical features from a breast cancer
dataset. The model learns patterns from labelled examples and predicts
one of two diagnosis classes:

-   **Benign**
-   **Malignant**

The notebook includes data inspection, preprocessing, exploratory data
analysis, model training, evaluation, and a sample prediction.

## Dataset

The notebook loads the CSV file named:

``` text
breast_cancer_wisconsin_dataset.csv
```

The dataset used in the saved notebook run has:

  Description                      Details
  ---------------- -----------------------
  Total records                        569
  Total columns                         31
  Input features                        30
  Target column                `diagnosis`
  Target classes     `benign`, `malignant`

The 30 numerical features describe characteristics such as radius,
texture, perimeter, area, smoothness, compactness, concavity, symmetry,
and fractal dimension. The features include mean, error, and worst-value
measurements.

The target column is `diagnosis`. The notebook uses `LabelEncoder` to
encode its class labels.

**Important:** The CSV file is required to run the notebook. Keep it in
the same directory as `cancer_prediction.ipynb`.

## Technologies Used

-   **Python** --- programming language
-   **Jupyter Notebook** --- development environment
-   **Pandas** --- data loading and manipulation
-   **NumPy** --- numerical operations
-   **Matplotlib** --- data visualization
-   **Seaborn** --- statistical visualization
-   **Scikit-learn** --- preprocessing, model training, and evaluation
-   **Joblib** --- imported in the notebook

## Project Workflow

1.  **Import libraries:** Load the Python libraries required for data
    processing, visualization, and machine learning.
2.  **Load the dataset:** Read the CSV file with Pandas.
3.  **Explore the data:** Check the dataset shape, column information,
    descriptive statistics, missing values, and duplicate rows.
4.  **Preprocess the data:**
    -   Remove duplicate rows.
    -   Separate the input features (`X`) from the target (`y`).
    -   Apply median imputation to the feature data.
    -   Encode the diagnosis labels using `LabelEncoder`.
5.  **Visualize the data:**
    -   Diagnosis class-count plot.
    -   Numerical feature correlation heatmap.
    -   Histograms for selected features.
    -   Boxplots for selected features.
6.  **Split the data:** Divide the data into training and testing sets.
7.  **Create and train the model:** Configure and fit a Gradient
    Boosting Classifier.
8.  **Make predictions:** Predict the diagnosis classes for the test
    data.
9.  **Evaluate the model:** Calculate accuracy, print a classification
    report, and display a confusion matrix.
10. **Compare predictions:** Display actual and predicted labels.
11. **Test a sample:** Run a prediction on the first row of the test set
    and display its class probabilities.

## Machine Learning Model

The project uses `GradientBoostingClassifier` from `sklearn.ensemble`.

The model is configured with the following parameters:

  Parameter           Value
  ----------------- -------
  `n_estimators`        100
  `learning_rate`      0.05
  `max_depth`             2
  `random_state`         42

The dataset is divided using `train_test_split` with:

  Parameter          Value
  ---------------- -------
  `test_size`         0.20
  `random_state`        42
  `stratify`           `y`

This produces 455 training records and 114 testing records in the saved
notebook run.

## Model Evaluation

The saved notebook run reports the following test-set results:

**Accuracy: 96.49%**

### Classification Report

  Class                    Precision   Recall   F1-score   Support
  ---------------------- ----------- -------- ---------- ---------
  Benign                        0.95     1.00       0.97        72
  Malignant                     1.00     0.90       0.95        42
  **Overall accuracy**                          **0.96**   **114**

The notebook also generates a confusion matrix to compare actual and
predicted classes.

These values describe the saved notebook run. Your results may vary if
the dataset or code is changed.

## Project Structure

Keep the notebook and dataset together in one folder:

``` text
Breast-Cancer-Prediction/
│
├── cancer_prediction.ipynb
├── breast_cancer_wisconsin_dataset.csv
└── README.md
```

## Installation

### 1. Install Python

Install Python 3 from the official website:

https://www.python.org/downloads/

### 2. Install the required libraries

Open Command Prompt or a terminal and run:

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

## How to Run

1.  Download or clone the project.

2.  Place `breast_cancer_wisconsin_dataset.csv` in the same folder as
    `cancer_prediction.ipynb`.

3.  Open a terminal or Command Prompt in the project folder.

4.  Start Jupyter Notebook:

    ``` bash
    jupyter notebook
    ```

5.  Open `cancer_prediction.ipynb` in your browser.

6.  Select **Kernel → Restart & Run All** (or run the cells from top to
    bottom).

7.  Review the visualizations, evaluation metrics, and sample
    prediction.

## Example Prediction

The notebook takes the first sample from the test set and predicts its
diagnosis using:

``` python
new_patient = X_test.iloc[[0]]

prediction = model.predict(new_patient)
prediction_name = encoder.inverse_transform(prediction)

probability = model.predict_proba(new_patient)[0]
```

In the saved notebook run, the sample prediction is:

``` text
New Patient Prediction: benign

benign : 98.63 %
malignant : 1.37 %
```

This is an example output from one test record, not a prediction for a
real patient's medical condition.
