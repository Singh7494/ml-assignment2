# Machine Learning Assignment 2

BITS Pilani | M.Tech (AIML)

## a. Problem Statement

Classify breast tumours as malignant or benign using five machine learning models. Evaluate each model using six metrics and provide a Streamlit application for evaluating and comparing saved models on uploaded test data.

## b. Dataset Description

| Property | Details |
| :--- | :--- |
| Dataset | Breast Cancer Wisconsin (Diagnostic) |
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), loaded through scikit-learn |
| Instances | 569 |
| Input features | 30 numeric features: mean, standard error and worst measurements of cell-nucleus characteristics |
| Target | 0 = Malignant; 1 = Benign |
| Missing values | None |
| Train/test split | Stratified 80/20 split: 455 training and 114 test instances; `random_state=42` |
| Preprocessing | StandardScaler fitted on training data within each saved model pipeline |
| Evaluation data | 114 held-out test instances |
| Positive class for precision, recall and F1 | Benign (1) |

## c. GitHub Repository Link

[GitHub Repository](https://github.com/YOUR_USERNAME/ml-assignment2)

## d. Models Used and Evaluation Results

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Joint-highest accuracy, F1 and MCC, with the highest AUC (0.9954). |
| Decision Tree | Lowest accuracy (0.9211) and MCC (0.8341) among the five models on this split. |
| KNN | Accuracy of 0.9561, ranking behind Logistic Regression and SVM; benign recall is 0.9722. |
| Naive Bayes | Lower accuracy than KNN, but higher AUC (0.9868 versus 0.9788). |
| Support Vector Machine | Ties Logistic Regression on five metrics; its AUC is slightly lower (0.9950). |
| Overall Winner for the Dataset | Logistic Regression, using AUC to break its tie with SVM on the other five metrics. |
