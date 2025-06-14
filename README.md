#  Breast Cancer Prediction Project
## Introduction

This project focuses on building and evaluating machine learning models to **Detection of Breast Cancer** based on diagnostic measurements. Early and accurate detection of breast cancer is crucial for successful treatment and improved patient outcomes. This analysis uses machine learning techniques to assist in this critical diagnostic process.

The primary goal is to develop a robust classification model that can accurately distinguish between benign and malignant tumours, thereby offering a valuable tool for medical professionals.

## Data Source

The dataset used in this project is the **Breast Cancer (Diagnostic) Dataset** 
* **Source:** Provided by Internpe (https://drive.google.com/drive/folders/1UXH9SKM4LA04lI0QnoktaD-jGyJL7w1j?usp=sharing)
* **Description:** This dataset contains 569 instances, 33 numeric features computed from a digitised image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the id, diagnosis, radius_mean, texture_mean,  perimeter_mean, area_mean,smoothness_mean, compactness_mean,  concavity_mean, concave points_mean, texture_worst, perimeter_worst, area_worst, smoothness_worst,compactness_worst, concavity_worst, concave points_worst, symmetry_worst,  fractal_dimension_worst, Unnamed: 32  . The target variable is `diagnosis`, indicating whether the tumor is `M` (Malignant) or `B` (Benign).

## Methodology

The project followed a standard data science pipeline:
1.  **Data Acquisition & Loading:** Loaded the dataset using Pandas.
2.  **Exploratory Data Analysis (EDA):**
    * Analyzed dataset summary statistics and feature distributions.
    * Identified relationships between features and the target variable (diagnosis).
    * Visualized data distributions and correlations to gain insights into feature importance.
3.  **Data Preprocessing:**
    * Handled missing values (though none were present in this specific dataset, this step is included for completeness).
    * Feature scaling using `StandardScaler` to normalise numerical features, which is crucial for distance-based algorithms.
    * Encoded the categorical target variable (M/B) into numerical format (0/1).
4.  **Model Selection & Training:**
    * Split the dataset into training and testing sets (80/20 ratio).
    * Evaluated several classification algorithms, including:
        * **Logistic Regression:** A good baseline linear model.
        * **Support Vector Machine (SVM):** Effective for high-dimensional data.
        * **Random Forest Classifier:** An ensemble method known for its robustness.
        * **Gradient Boosting (XGBoost):** Another powerful ensemble method.
    * Trained each model on the preprocessed training data.
5.  **Model Evaluation:**
    * Assessed model performance on the unseen test set using various metrics:
        * **Accuracy Score:** Overall correctness.
        * **Precision:** Proportion of true positive predictions among all positive predictions.
        * **Recall (Sensitivity):** Proportion of true positive predictions among all actual positives. Crucial for medical diagnosis to minimize false negatives.
        * **F1-Score:** Harmonic mean of precision and recall.
        * **ROC AUC Score:** Measures the model's ability to distinguish between classes.
        * **Confusion Matrix:** Visualizes the true and false positives/negatives.
6.  **Hyperparameter Tuning (for best performing models):** Used techniques like GridSearchCV to optimize model parameters.
## Tools and Technologies

7. **Language:** Python 3.12
8. **Libraries:**
    * `pandas` for data manipulation and analysis
    * `numpy` for numerical operations
    * `scikit-learn` for machine learning models, preprocessing, and evaluation
    * `matplotlib.pyplot` for basic plotting
    * `seaborn` for enhanced statistical data visualization
* **Environment:** Jupyter Notebook on PyCharm

## Key Findings and Model Performance

After rigorous evaluation, the **Random Forest Classifier** emerged as the top-performing model, achieving excellent results in predicting breast cancer.

* **Best Model:** Random Forest Classifier
* **Key Performance Metrics (on test set):**
    * **Accuracy:** ~97.36%
    * **Precision (Malignant):** ~97.97%
    * **Recall (Malignant):** ~96.06%
    * **F1-Score (Malignant):** ~97.06%
    * **ROC AUC Score:** ~0.97

The high recall score for malignant cases is particularly important in a medical context, as minimizing false negatives (missing actual cancer cases) is paramount. The model demonstrates strong generalization capabilities, indicating its potential utility.

**Important Features:** Feature importance analysis revealed that `concave_points_mean`, `texture_mean`, `fractal_dimension_worst`, `radius_mean `, and `concavity_mean` were among the most influential features in predicting breast cancer.

## Visualizations

Here are some key visualizations from the EDA and model evaluation phases:

### Distribution of Diagnosis
![Diagnosis Distribution](images/diagnosis_distribution.png)
* This plot shows the count of benign (B) and malignant (M) diagnoses in the dataset, indicating a slight imbalance.

### Correlation Heatmap of Features
![Correlation Heatmap](images/correlation_heatmap.png)
* A heatmap illustrating the correlation between various features. Highly correlated features might indicate multicollinearity and potential redundancy.

### ROC Curve for Best Model
![ROC Curve](images/roc_curve.png)
* The Receiver Operating Characteristic (ROC) curve for the Random Forest Classifier. The area under the curve (AUC) close to 1.0 indicates excellent discriminative power.

### Confusion Matrix for Best Model
![Confusion Matrix](images/confusion_matrix.png)
* A confusion matrix for the Random Forest Classifier, showing the counts of true positives, true negatives, false positives, and false negatives.

---
