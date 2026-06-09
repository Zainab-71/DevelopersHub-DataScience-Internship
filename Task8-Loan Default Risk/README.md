# Task 4: Loan Default Risk with Business Cost Optimization

## Objective

Predict the likelihood of a loan default using binary classification models, and optimize the decision threshold based on a cost-benefit analysis to minimize the total financial cost of misclassification.

## Dataset

**Home Credit Default Risk Dataset**

- Features include credit score, income, loan amount, interest rate, loan type, loan purpose, LTV (loan-to-value ratio), term, upfront charges, and age
- Target variable: `Status` — whether a loan applicant defaulted (1) or not (0)
- The dataset has a class imbalance, with non-defaulters significantly outnumbering defaulters

## Approach

- Loaded and inspected the dataset for shape, column types, and class distribution
- Visualized default distribution, gender vs default, credit score distribution, income distribution, and loan amount vs default status
- Selected 13 relevant features and dropped columns with no predictive value
- Filled missing numerical values with the column median and label encoded all categorical columns
- Split data into 80/20 train-test sets using stratified sampling to preserve class ratios
- Trained Logistic Regression and Random Forest classifiers; Random Forest used `class_weight='balanced'` to address class imbalance
- Evaluated both models using Confusion Matrix, F1-Score, and ROC-AUC Curve
- Defined business cost values — $500 for approving a defaulter (false positive) and $100 for rejecting a good customer (false negative)
- Swept decision thresholds from 0.1 to 0.9 on the Random Forest probability outputs to find the threshold that minimizes total business cost
- Extracted Random Forest feature importances to identify the strongest predictors of default

## Results and Insights

- Random Forest outperformed Logistic Regression with a higher F1-Score and AUC
- The default 0.5 threshold was not cost-optimal — the cost analysis identified a lower threshold that significantly reduced total business cost by catching more defaulters earlier
- **Top predictors of loan default:** `Credit_Score`, `LTV`, `rate_of_interest`, `loan_amount`, and `income`
- Lower credit scores and higher LTV ratios were consistently associated with higher default risk
- Higher interest rates and lower income also pushed predictions toward default

## Business Recommendation

Use the cost-optimized threshold in production rather than the standard 0.5 cutoff. Prioritise applicants with high credit scores and low LTV ratios. Flag applications with high interest rates, high loan amounts relative to income, or unknown credit worthiness for additional review before approval. Avoid approving borderline applicants without manual verification, as the cost of a missed default ($500) is five times the cost of rejecting a good customer ($100).

## Tools and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (Logistic Regression, Random Forest, ROC Curve, Confusion Matrix)
**Zainab**
Data Science & Analytics Intern – DevelopersHub Corporation
[GitHub](https://github.com/Zainab-71)
