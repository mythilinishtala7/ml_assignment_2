# Machine Learning Assignment 2  
## Income Classification using Multiple ML Models

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related attributes.

This is a binary classification problem where:

- Class 0 → Income ≤ 50K  
- Class 1 → Income > 50K  

Six different machine learning models were implemented and evaluated using multiple performance metrics to compare their effectiveness.


## b. Dataset Description

The dataset used in this project is the Adult Income Dataset (Census Income Dataset) from the UCI Machine Learning Repository.

### Dataset Characteristics:

- Total Instances: 48,842  
- Number of Original Features: 14  
- Target Variable: income  
- Type: Binary Classification  

### Feature Types:

Numerical Features:
- age  
- fnlwgt  
- education-num  
- capital-gain  
- capital-loss  
- hours-per-week  

Categorical Features:
- workclass  
- education  
- marital-status  
- occupation  
- relationship  
- race  
- sex  
- native-country  


### Preprocessing Steps Performed:

1. Replaced missing values marked as "?" with NaN  
2. Removed rows containing missing values  
3. Converted target variable into binary format (0 and 1)  
4. Applied One-Hot Encoding for categorical variables  
5. Applied StandardScaler for feature scaling  
6. Performed 80–20 Train-Test split  


## c. Models Used

The following six classification models were implemented:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (kNN)  
4. Gaussian Naive Bayes  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

## Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Logistic Regression | 0.8572 | 0.9107 | 0.7371 | 0.6074 | 0.6660 | 0.5807 |
| Decision Tree | 0.8635 | 0.9047 | 0.7657 | 0.6022 | 0.6742 | 0.5962 |
| kNN | 0.8291 | 0.8445 | 0.6496 | 0.5878 | 0.6171 | 0.5085 |
| Naive Bayes | 0.5690 | 0.8146 | 0.3470 | 0.9507 | 0.5084 | 0.3560 |
| Random Forest | 0.8676 | 0.9203 | 0.7958 | 0.5856 | 0.6747 | 0.6051 |
| XGBoost | 0.8794 | 0.9313 | 0.7936 | 0.6563 | 0.7185 | 0.6473 |


## Model Observations

### Logistic Regression
Logistic Regression performed strongly with a high AUC score (0.9107), indicating good probability estimation. It achieved balanced precision and recall after proper feature scaling.

### Decision Tree
Decision Tree captured non-linear relationships effectively and achieved slightly better accuracy than Logistic Regression. However, recall remained moderate, suggesting limited ability to detect all high-income individuals.

### k-Nearest Neighbors (kNN)
kNN showed reasonable performance but lower overall metrics compared to tree-based and boosting models. The high dimensionality after one-hot encoding likely impacted its performance.

### Naive Bayes
Naive Bayes achieved very high recall (0.9507), meaning it identified most high-income individuals. However, its precision was very low, leading to poor overall accuracy and MCC. This reflects the strong independence assumption of Naive Bayes.

### Random Forest (Ensemble)
Random Forest improved performance over a single Decision Tree. It achieved better MCC and AUC, demonstrating reduced overfitting and improved generalization through ensemble averaging.

### XGBoost (Ensemble)
XGBoost achieved the best overall performance across almost all metrics, including highest Accuracy (0.8794), AUC (0.9313), F1 score (0.7185), and MCC (0.6473). Its boosting mechanism allowed it to effectively capture complex feature interactions and deliver superior classification performance.


## Conclusion

Among all models, XGBoost performed the best, followed by Random Forest and Decision Tree. Ensemble methods demonstrated superior performance compared to individual models due to their ability to reduce variance and improve generalization.

## Deployment

The Streamlit application provides:

- CSV upload functionality (test dataset)  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  
