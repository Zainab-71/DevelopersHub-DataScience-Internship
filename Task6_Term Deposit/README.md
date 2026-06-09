# Task 6: Term Deposit Subscription Prediction (Bank Marketing)

## Objective

Predict whether a bank customer will subscribe to a term deposit as a result of a marketing campaign, and identify which customer segments are most likely to respond positively.

## Dataset

**Bank Marketing Dataset – UCI Machine Learning Repository**

- 45,211 customer records and 17 columns
- Features include age, job, marital status, education, account balance, call duration, contact type, month, and previous campaign outcomes
- Target variable: `y` — whether the customer subscribed to a term deposit (yes / no)
- The dataset has a class imbalance, with significantly more `no` responses than `yes`

## Approach

- Loaded and inspected the dataset for missing values and class distribution
- Visualized key features — subscription distribution, age, job type, and account balance — against the target variable
- Encoded all categorical features using Label Encoding
- Split data into 80/20 train-test sets
- Trained Logistic Regression and Random Forest classifiers
- Evaluated both models using Confusion Matrix, F1-Score, and ROC-AUC Curve
- Applied SHAP (TreeExplainer) on the Random Forest to explain feature-level contributions to individual predictions

## Results and Insights

- Random Forest outperformed Logistic Regression with a higher F1-Score and AUC
- `duration` (call duration) was the strongest predictor — longer calls significantly increased subscription probability
- Students and retired customers had higher acceptance rates compared to blue-collar workers
- Customers with higher account balances were more likely to subscribe
- Repeated campaign contacts (high `campaign` values) were associated with lower subscription rates
- July, August, and unknown contact methods consistently pushed predictions toward `no`

## Business Recommendation

Prioritise customers with longer call engagement and higher account balances. Target retired customers and students in future campaigns, and avoid excessive repeated contact — diminishing returns appear quickly. Run campaigns in months that historically show higher conversion, and focus on customers with a positive response history from previous campaigns.

## Tools and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP

**Zainab**
Data Science & Analytics Intern – DevelopersHub Corporation
[GitHub](https://github.com/Zainab-71)
