# BİL 3013 - Data Mining Coursework

This repository contains the assignments and projects developed for the **BİL 3013 Introduction to Data Mining** course at **Dokuz Eylul University** (Computer Science, Fall 2025).

The repository showcases the implementation of various data mining techniques including **Unsupervised Learning (Clustering)**, **Regression Analysis**, and **Supervised Learning (Classification)** on real-world and custom datasets.

**Course Instructor:** Prof. Dr. Efendi NASİBOĞLU

## 👥 Contributors
* **[Salim Taha KAVAS](https://github.com/stobubul)**
* **[İremgül ZEYTİNÖZÜ](https://github.com/iremzeytinozu)**

---

## 📂 Repository Contents

### 1. Assignment 1: Clustering (Breast Cancer Analysis)
**Goal:** Applying unsupervised learning algorithms to detect patterns in the **Wisconsin Breast Cancer (Diagnostic)** dataset without using label information.

* **Techniques Used:**
    * **Data Preprocessing:** Standardization (`StandardScaler`) to handle scale differences between features like area and smoothness.
    * **Dimensionality Reduction:** PCA (Principal Component Analysis) to visualize 30-dimensional data in 2D.
    * **Algorithms:** K-Means Clustering and DBSCAN.
    * **Optimization:** Elbow Method and Silhouette Analysis for determining optimal *K*.
* **Key Results:**
    * **K-Means (K=2)** significantly outperformed DBSCAN with an Adjusted Rand Score (ARI) of **0.6765**, successfully separating Benign and Malignant cases.
    * DBSCAN failed to capture the global structure, classifying 39% of data as noise.

### 2. Assignment 2: Regression Models (Student Performance)
**Goal:** Predicting student midterm scores based on weekly study hours using a custom dataset from the "BİL 2009 Graph Theory" course.

* **Techniques Used:**
    * **Algorithms:** Linear Regression, Decision Tree Regressor, Support Vector Regression (SVR).
    * **Evaluation Metrics:** $R^2$ Score, MAE, RMSE.
    * **Correlation Analysis:** Found a positive correlation (*r=0.63*) between study hours and grades.
* **Key Results:**
    * **Decision Tree Regressor** achieved the best performance ($R^2 = 0.547$), capturing the non-linear "threshold" effect where grades spike after 1.5 hours of study.
    * Linear Regression analysis suggested that **1 hour of extra study** increases the score by approximately **11.02 points**.

### 3. Assignment 3: Classification (Credit Card Fraud Detection)
**Goal:** Detecting fraudulent transactions in a highly imbalanced dataset (Kaggle Credit Card Fraud Detection).

* **Techniques Used:**
    * **Handling Imbalance:** Used `RobustScaler` and `class_weight='balanced'` strategies.
    * **Algorithms:** Random Forest, XGBoost, and MLP (Artificial Neural Networks).
    * **Feature Importance:** Identified V14 and V4 as the most critical features for fraud detection.
* **Key Results:**
    * **Random Forest** proved to be the most balanced model with the highest **Precision (0.897)** and an **F1-Score of 0.843**, minimizing false alarms.
    * **XGBoost** provided the highest Recall (catching more fraud) but with a higher false positive rate.

---

## 🛠 Technologies & Tools
The projects were developed using **Python** in **Jupyter Notebook** environments.

* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (sklearn), XGBoost
* **Visualization:** Matplotlib, Seaborn
* **Preprocessing:** StandardScaler, RobustScaler, PCA

---

### 📜 Disclaimer
This repository is created for educational purposes as part of the coursework for BİL 3013 at Dokuz Eylul University.
