# Task 3: Customer Churn Prediction (Bank Customers)

## Objective
Build a machine learning model that predicts whether a bank customer is likely to leave (churn), 
based on personal and financial information such as age, credit score, balance, and product usage.

## Approach
- Inspected each column individually before deciding how to handle the data.
- Dropped irrelevant identifier columns: RowNumber, CustomerId, and Surname.
- Applied Label Encoding to Gender (2 unique values) and One-Hot Encoding to Geography 
  (3 unique values: France, Germany, Spain).
- Visualized churn distribution, age, balance, and product usage patterns.
- Trained a Random Forest Classifier and evaluated it using accuracy score, confusion matrix, 
  and classification report.
- Extracted and ranked feature importances to identify what drives churn.

## Results and Insights
- Age was the strongest predictor of churn with an importance score of 0.24, every other 
  feature scored below 0.15.
- Churned customers were noticeably older, with a median age around 45 versus 35 for retained customers.
- Customers with 3 or 4 products churned at nearly 100%, making product count a critical warning signal.
- Random Forest achieved 86.60% accuracy and outperformed the baseline, but recall for the 
  churned class was only 0.46 due to class imbalance.
- Both the confusion matrix and classification report confirmed the model is much better at 
  identifying customers who stay than those who leave.
- This happened because the dataset contains roughly 8,000 retained customers and only 2,000 
  churned ones. The model saw far more examples of staying during training, so it became better 
  at predicting retention. It struggled to recognize churn simply because it did not see enough examples of it.
- A larger and more balanced dataset would be needed before deploying this model in a real 
  customer retention environment.

## Tools and Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
