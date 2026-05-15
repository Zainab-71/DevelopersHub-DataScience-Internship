## Task 2: Credit Risk Prediction

### Objective
Build a machine learning model that predicts whether a loan applicant is likely to repay their loan or default, based on 
personal and financial information such as income, education, credit history, and loan amount.

### Approach
- Inspected each column individually before deciding how to handle missing values.
- Used mode for text and binary columns, median for skewed numeric columns.
- Encoded all text columns into numbers for ML compatibility.
- Visualized loan amount, income, education, and credit history distributions.
- Trained two classification models: Logistic Regression and Decision Tree.
- Evaluated both models using accuracy score, confusion matrix, and classification report.

### Results and Insights
- Credit History was the strongest predictor of loan approval with a correlation of 0.54 — every other feature stayed below 0.10.
- Applicants with bad credit were almost universally rejected while those with good credit were approved at a very high rate.
- Loan amount and applicant income were both right-skewed, a small number of high values pulled the averages up significantly.
- Logistic Regression achieved 79.67% accuracy and outperformed Decision Tree which achieved 73.98%
- Both models struggled to correctly identify rejections. Only 18 out of 40 actual rejections were caught by Logistic Regression
- This happened because the dataset contains roughly 420 approved loans and only 190 rejected ones. The model practiced predicting
  approvals far more than rejections during training, so it became much better at spotting approvals. It struggled to recognize
  rejections simply because it did not see enough examples of them.
- A larger and more balanced dataset would be needed before deploying this model in a real lending environment

### Tools and Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
