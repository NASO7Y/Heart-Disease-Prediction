# Heart-Disease-Prediction
This project uses machine learning to predict heart disease from a dataset (heart_disease.csv). The data was preprocessed with label encoding, imputation, and SMOTE for class balancing. Models like SVM, KNN, Random Forest, and Decision Tree were applied and evaluated using accuracy, confusion matrices, and classification reports.


Here is the details :

---

### Heart Disease Prediction Using Machine Learning

This project aims to predict heart disease using various machine learning algorithms, based on a dataset (`heart_disease.csv`). The data was preprocessed, and multiple classifiers were applied to evaluate the best model for predicting heart disease.

#### Steps:
1. **Data Preprocessing:**
   - The dataset was cleaned by handling missing values using the `SimpleImputer` (mean imputation).
   - Categorical columns were encoded using `LabelEncoder`.
   - Duplicate rows were removed, and the dataset was reset.
   - Class imbalance was handled using `SMOTE` to oversample the minority class.

2. **Exploratory Data Analysis (EDA):**
   - Various visualizations were created, including:
     - Pie charts to show the distribution of heart disease based on gender and fasting blood sugar.
     - Histograms to visualize feature distributions.
     - A correlation heatmap to identify relationships between features.

3. **Feature Scaling:**
   - Min-Max scaling was applied to normalize the data.

4. **Modeling:**
   - Several machine learning models were applied:
     - **Support Vector Machine (SVM)**: Tuned using `GridSearchCV` for optimal parameters.
     - **K-Nearest Neighbors (KNN)**: Hyperparameters optimized using `GridSearchCV` and cross-validation.
     - **Random Forest Classifier (RFC)**: Fitted with the training data.
     - **Decision Tree Classifier (DTC)**: Applied with the "entropy" criterion.

5. **Model Evaluation:**
   - Models were evaluated using accuracy scores and confusion matrices.
   - Performance metrics (precision, recall, F1-score) were displayed using `classification_report` for each model.

6. **Results:**
   - The SVM model, KNN, Random Forest, and Decision Tree all demonstrated good performance in predicting heart disease.
   - Confusion matrices were visualized for each model to assess their ability to correctly classify heart disease cases.

#### Conclusion:
The project successfully applied multiple machine learning models to predict heart disease and evaluated their performance. The data preprocessing, feature scaling, and class balancing techniques contributed to the models' effectiveness.

---
