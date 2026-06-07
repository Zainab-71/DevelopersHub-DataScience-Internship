## Task 5: Personal Loan Acceptance Prediction

### Objective
Predict whether a bank customer will accept a personal loan 
offer and identify which customer groups the bank should 
prioritize in future marketing campaigns.

### Dataset
Bank Marketing Dataset — 11,162 customer records and 17 columns 
including age, job, marital status, education, account balance, 
call duration, and previous campaign outcomes. Target variable 
is deposit (yes or no). The dataset is well balanced with 5,289 
yes and 5,873 no responses.

### Approach
- Inspected all columns before encoding — binary text columns 
  were label encoded, multi-value columns (job, marital, 
  education, contact, month, poutcome) were one-hot encoded
- Preserved original dataframe before encoding to allow 
  business insight charts to display readable labels
- Visualized age, call duration, balance, and job type 
  against acceptance rate
- Calculated acceptance rates by job type, marital status, 
  and age group to identify which customer segments to target
- Trained Logistic Regression and Decision Tree classifiers
- Extracted model coefficients to understand which features 
  drive acceptance most strongly

### Results and Insights
- Logistic Regression achieved 81.10% and outperformed 
  Decision Tree at 78.91%
- Call duration had the strongest numeric correlation at 0.45 
  — yes customers talked nearly twice as long as no customers
- Students and retired customers had the highest acceptance 
  rates at 75% and 65% respectively
- Customers aged 60 and above accepted at over 80% — highest 
  of any age group
- Previous campaign success was the strongest model driver 
  — customers who said yes before are very likely to say yes again
- March, October, and September were the best months for campaigns
- July and August and unknown contact methods consistently 
  pushed predictions toward no

### Business Recommendation
Prioritize retired customers and students, re-target previous 
campaign responders, and run campaigns in March, October, and 
September. Avoid calling customers with unknown contact details 
or existing personal loans during summer months.

### Tools and Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
