# Telco Customer Churn Prediction

A complete end-to-end machine learning project for predicting customer churn using a Telco-style customer churn dataset. The project covers exploratory data analysis, data preprocessing, Random Forest model building, class imbalance handling, cross-validation, ROC-AUC evaluation, feature importance analysis, and customer segmentation using K-Means clustering.

This repository follows the same practical churn-analysis workflow as a typical Telco churn project, with a clean synthetic dataset created for learning, portfolio, and demo use.

---

## Dataset

- **File:** `Telco_customer_churn.xlsx`
- **Sheet:** `CustomerData`
- **Target Variable:** `Churn Value` (`1` = Churned, `0` = Retained)
- **Key Features:** `Tenure Months`, `Monthly Charges`, `Total Charges`, `Contract`, `Internet Service`, `Payment Method`, `Tech Support`, `Online Security`, `Churn Score`, and `CLTV`.

The dataset is synthetic but shaped like a real telecom churn dataset, so the notebooks can be run end to end without depending on a third-party data license.

---

## Files

| File | Description |
| --- | --- |
| `Telco_customer_churn.xlsx` | Synthetic Telco-style customer churn dataset. |
| `CBSOTPROJ1.ipynb` | Main notebook with EDA, cleaning, modeling, ROC-AUC, feature importance, and K-Means segmentation. |
| `CBSOT_SIP_Customer_Churn_Analysis.ipynb` | Business-focused notebook with churn insights, risk bands, and retention recommendations. |
| `requirements.txt` | Python dependencies required to run the notebooks. |

---

## Libraries Used

```python
pandas, numpy, matplotlib, seaborn
sklearn (RandomForestClassifier, KMeans, StandardScaler, train_test_split, metrics)
openpyxl
```

---

## Step 1: Exploratory Data Analysis (EDA)

The EDA phase studies customer churn behavior across tenure, billing, contract type, service usage, and support-related fields.

- **Churn Distribution:** Checked `Churn Label` and `Churn Value` to understand the retained vs churned customer split.
- **Tenure Months:** Plotted tenure distribution and compared churned vs retained customers.
- **Monthly Charges:** Compared billing patterns for churned and non-churned customers.
- **Contract Type:** Reviewed churn concentration across month-to-month, one-year, and two-year contracts.
- **Internet Service:** Compared churn for DSL, fiber optic, and no internet service customers.
- **Payment Method:** Analyzed churn rate differences across payment options.
- **Tech Support:** Checked whether customers without support services show higher churn.
- **Correlation Matrix:** Reviewed relationships across numerical columns such as tenure, charges, churn score, churn value, and CLTV.
- **Cross-Tabulation:** Used normalized crosstabs to calculate churn rate by contract and service category.

---

## Step 2: Data Cleaning

- **`Total Charges` conversion:** Converted `Total Charges` to numeric using `pd.to_numeric(errors="coerce")`.
- **Missing values:** Filled missing or invalid total charges with `0`, which handles zero-tenure customer records.
- **Dropping irrelevant columns:** Removed identifiers, geography fields, redundant fields, and target-leakage fields before modeling.

```python
drop_columns = [
    "CustomerID", "Count", "Country", "State", "Zip Code",
    "Lat Long", "Latitude", "City", "Longitude",
    "Churn Label", "Churn Score", "CLTV", "Churn Reason"
]
```

---

## Step 3: Encoding

- One-hot encoding is applied to categorical columns using `pd.get_dummies(..., drop_first=True)`.
- Text categories such as contract type, payment method, and internet service are converted into numeric model-ready columns.
- The target column `Churn Value` is kept separate from the feature matrix.

---

## Step 4: Feature Selection

- **X (Features):** All cleaned and encoded columns except `Churn Value`.
- **Y (Target):** `Churn Value`, where `1` represents churn.
- Feature importance scores are extracted from the tuned Random Forest model.
- Low-importance columns are reviewed and removed to test whether a smaller feature set can keep similar performance.

```python
feature_importance = pd.DataFrame({
    "Feature": X_encoded.columns,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=False)
```

---

## Step 5: Train and Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

- **80% training / 20% testing** split is used.
- `random_state=42` keeps results reproducible.
- `stratify=y` keeps churn distribution balanced across train and test sets.

---

## Step 6: Random Forest - Three Approaches

### Baseline Model

```python
rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
```

The baseline model is evaluated using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC.

### Approach 1 - Handling Class Imbalance

```python
rf_balanced = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)
```

`class_weight="balanced"` gives more importance to churned customers, which helps improve recall for churn detection.

### Approach 2 - Hyperparameter Tuning

```python
rf_tuned = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=4,
    random_state=42,
    class_weight="balanced"
)
```

A manual scan compares multiple values of `n_estimators` and `max_depth`. Results are ranked by recall first, then ROC-AUC and accuracy.

### Approach 3 - Feature Importance Analysis

- Top features are ranked using Random Forest importance scores.
- Bottom features are reviewed for removal.
- A selected-feature model is trained to check whether performance remains stable with fewer columns.

---

## Step 7: Cross-Validation

```python
cv_accuracy = cross_val_score(final_rf, X_encoded, y, cv=5, scoring="accuracy")
cv_recall = cross_val_score(final_rf, X_encoded, y, cv=5, scoring="recall")
cv_auc = cross_val_score(final_rf, X_encoded, y, cv=5, scoring="roc_auc")
```

- **5-Fold Cross-Validation** gives a more reliable estimate of model performance.
- Accuracy, recall, and ROC-AUC are compared across folds.
- Recall is treated as an important metric because the goal is to catch likely churners early.

---

## Step 8: ROC-AUC Curve

```python
y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)
```

- `predict_proba` is used to generate churn probability scores.
- ROC curve compares false positive rate vs true positive rate.
- AUC score measures how well the model separates churned and retained customers.

---

## Step 9: Customer Segmentation using K-Means

Churn probability is combined with billing and tenure features to create customer segments for retention planning.

### Segmentation Features

```python
segmentation_data = pd.DataFrame({
    "Tenure Months": working_df["Tenure Months"],
    "Monthly Charges": working_df["Monthly Charges"],
    "Total Charges": working_df["Total Charges"],
    "Churn Probability": churn_probability
})
```

### Scaling

```python
scaler = StandardScaler()
scaled_data = scaler.fit_transform(segmentation_data)
```

### Finding Optimal K

```python
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
```

The elbow method is used to review WCSS values and choose a practical number of clusters.

### Segment Interpretation

| Segment Type | Meaning |
| --- | --- |
| Low Risk Customers | Longer tenure and lower churn probability. |
| Medium Risk Customers | Mixed profile with moderate churn probability. |
| High Risk Customers | Shorter tenure, higher charges, and higher churn probability. |

---

## Step 10: Data Visualisation

All visualisations are created with Matplotlib and Seaborn.

| Chart | Purpose |
| --- | --- |
| Churn Countplot | Shows retained vs churned customer count. |
| Tenure Histplot | Shows distribution of customer tenure. |
| Tenure Boxplot | Compares tenure for churned and retained customers. |
| Monthly Charges Histplot | Reviews monthly billing distribution. |
| Monthly Charges Boxplot | Compares billing by churn status. |
| Contract Countplot | Shows churn by contract type. |
| Internet Service Countplot | Compares churn across service types. |
| Payment Method Countplot | Reviews churn by payment method. |
| Tech Support Countplot | Shows impact of support availability. |
| Correlation Heatmap | Reviews relationships between numeric variables. |
| ROC Curve | Measures model discrimination. |
| Elbow Curve | Helps choose K-Means cluster count. |
| Segment Scatter Plot | Visualizes churn probability against tenure and charges. |

---

## How to Run

1. Clone or download this repository.
2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Keep `Telco_customer_churn.xlsx` in the same folder as the notebooks.
4. Open `CBSOTPROJ1.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.
5. Run all cells from top to bottom.

---

## Key Takeaways

- Customers on month-to-month contracts with shorter tenure are more likely to churn.
- Higher monthly charges can increase churn risk, especially when customers lack support services.
- `class_weight="balanced"` helps improve recall for churned customers.
- ROC-AUC provides a better view of model separation than accuracy alone.
- K-Means segmentation converts model scores into actionable customer groups for retention planning.
