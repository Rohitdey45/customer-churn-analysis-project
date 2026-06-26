# Telco Customer Churn Prediction

A complete end-to-end machine learning project for predicting customer churn using a Telco-style customer churn dataset. The project covers exploratory data analysis, data preprocessing, Random Forest model building, recall-focused tuning, cross-validation, ROC-AUC evaluation, feature importance analysis, and customer segmentation using K-Means clustering.

This repository uses a synthetic Telco-style dataset created for learning and portfolio use. It mirrors common telecom churn fields such as tenure, contract type, monthly charges, total charges, internet service, payment method, tech support, churn label, churn score, and customer lifetime value.

## Files

| File | Description |
| --- | --- |
| `Telco_customer_churn.xlsx` | Synthetic customer churn dataset used by both notebooks. |
| `CBSOTPROJ1.ipynb` | Main analysis notebook with EDA, cleaning, model training, model comparison, ROC-AUC, and segmentation. |
| `CBSOT_SIP_Customer_Churn_Analysis.ipynb` | Compact business-focused notebook for churn insights, segment profiling, and retention recommendations. |
| `requirements.txt` | Python dependencies needed to run the notebooks. |

## Project Workflow

1. Load the Excel dataset and review customer-level churn fields.
2. Explore churn patterns by tenure, monthly charges, contract, internet service, payment method, and tech support.
3. Clean numeric fields and remove identifiers, geography fields, and target-leakage columns.
4. Encode categorical variables with one-hot encoding.
5. Train baseline, class-balanced, and tuned Random Forest models.
6. Compare accuracy, recall, precision, F1-score, confusion matrix, and ROC-AUC.
7. Review feature importance and test a smaller feature-selected model.
8. Build K-Means customer segments using tenure, charges, total charges, and churn probability.
9. Translate model findings into retention actions.

## Dataset

- `Churn Value`: target column, where `1` means churned and `0` means retained.
- Important predictors: `Tenure Months`, `Monthly Charges`, `Total Charges`, `Contract`, `Internet Service`, `Payment Method`, `Tech Support`, and `Online Security`.
- The data is synthetic, so it is safe to use for practice, demos, and portfolio work without depending on a third-party workbook license.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `CBSOTPROJ1.ipynb` first and run the cells from top to bottom. The notebook expects `Telco_customer_churn.xlsx` to be in the same folder.

## Key Takeaways

- Month-to-month customers with low tenure and higher monthly charges are usually the highest churn-risk group.
- Lack of support services such as tech support and online security is a strong churn signal.
- Recall is more important than raw accuracy when the business goal is to identify customers likely to leave.
- Customer segmentation helps convert churn probabilities into practical retention groups.
