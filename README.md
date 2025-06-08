# 📊 Customer Churn Analysis in SAS

This project performs **exploratory data analysis**, **feature engineering**, **logistic regression modeling**, and **model evaluation** using a **customer churn dataset** in SAS. The code also implements a manual **5-fold cross-validation** for robust performance assessment.

---

## 📁 Dataset: `customer_churn.csv`

The dataset contains records for 7,043 customers of a telecom company. The goal is to predict whether a customer is likely to **churn** (i.e., stop using the service).

### ✅ Key Columns:

| Column                                | Description                                       |
| ------------------------------------- | ------------------------------------------------- |
| `customerID`                          | Unique customer identifier                        |
| `gender`                              | Gender of the customer                            |
| `SeniorCitizen`                       | Whether the customer is a senior citizen (0 or 1) |
| `Partner`                             | Whether the customer has a partner                |
| `Dependents`                          | Whether the customer has dependents               |
| `tenure`                              | Number of months the customer has stayed          |
| `PhoneService`                        | Whether the customer has phone service            |
| `MultipleLines`                       | Whether the customer has multiple lines           |
| `InternetService`                     | Type of internet service                          |
| `OnlineSecurity`, `TechSupport`, etc. | Additional service options                        |
| `Contract`                            | Type of contract (Month-to-month, One year, etc.) |
| `PaymentMethod`                       | Billing method                                    |
| `MonthlyCharges`                      | Monthly charge amount                             |
| `TotalCharges`                        | Total charge amount (string column)               |
| `Churn`                               | Target variable: "Yes" or "No"                    |

---

## 🛠️ Project Pipeline

### 1. **Data Import & Inspection**

* Import the CSV into SAS using `PROC IMPORT`
* Check structure and missing values with `PROC CONTENTS` and `PROC MEANS`

### 2. **Data Cleaning & Transformation**

* Use `PROC MI` to impute missing values in `tenure`, `MonthlyCharges`, and `TotalCharges`
* Convert `TotalCharges` from character to numeric
* Create binary target column `churn_flag` (1 for churned, 0 for not)

### 3. **Feature Engineering**

* Create `tenure_group` (e.g., "0-12 mo", "12-24 mo")
* Create `MonthlyCharges_group` (Low, Medium, High)

### 4. **Multicollinearity Check**

* Use `PROC REG` with VIF to detect multicollinearity among numeric predictors

### 5. **Logistic Regression Modeling**

* Use `PROC LOGISTIC` to model churn
* Include categorical predictors with `param=ref` for reference coding

### 6. **Model Evaluation**

* Output predicted probabilities
* Generate binary predictions using a 0.5 threshold
* Build a **confusion matrix**
* Calculate **Precision**, **Recall**, and **F1-Score**

### 7. **ROC Curve**

* Generate a ROC curve using `plots=roc` option in `PROC LOGISTIC`

### 8. **5-Fold Cross-Validation**

* Manually split data into 5 folds
* Use macro `%kfold_logit()` to train/test on each fold
* Evaluate aggregated confusion matrix across folds

---

## 📈 Output Summary

* Classification performance metrics: Confusion Matrix, F1 Score
* Visual evaluation: ROC Curve
* Cross-validation: Model robustness across folds

---

## 📦 Files Included

| File                 | Description                       |
| -------------------- | --------------------------------- |
| `customer_churn.csv` | Input dataset                     |
| `churn_analysis.sas` | Complete SAS code for analysis    |
| `README.md`          | Project documentation (this file) |

---

## 🧠 Insights You Can Gain

* Which customer features most influence churn?
* How well can we predict churn using logistic regression?
* How robust is the model when evaluated using k-fold cross-validation?

---

## 📚 Tools Used

* **SAS Base**: Data manipulation, analysis
* **PROC IMPORT / MEANS / FREQ / MI / LOGISTIC / REG**
* Manual Macro-based k-fold cross-validation

---

## 🧩 Future Improvements

* Use advanced models (e.g., decision trees, SVMs) in Python/R
* Deploy model using SAS Viya or Python-based REST APIs
* Automate imputation using multiple strategies (mean/mode/ML)

---
