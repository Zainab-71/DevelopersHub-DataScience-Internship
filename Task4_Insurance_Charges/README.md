## Task 4: Predicting Insurance Claim Amounts

### Objective
Estimate medical insurance charges based on a customer's personal 
details such as age, BMI, smoking status, number of children, 
and region.

### Dataset
Medical Cost Personal Dataset from Kaggle — 1,338 records with 
7 columns. No missing values were found. Target variable is 
charges — an actual dollar amount, not a category.

### Approach
- Inspected all columns before encoding — sex and smoker had 
  2 unique values each so label encoding was used, region had 
  4 unique values so one-hot encoding was applied to avoid 
  creating a false numerical ranking between regions
- Visualized how age, BMI, and smoking status individually and 
  together affect insurance charges
- Trained a Linear Regression model to predict charge amounts
- Evaluated performance using MAE and RMSE instead of accuracy 
  since this is a regression problem — the output is a number, 
  not a category
- Plotted actual vs predicted charges and a residual plot to 
  visually assess model performance

### Results and Insights
- Smoking status was the strongest predictor with a correlation 
  of 0.79 against charges — far above every other feature
- Being a smoker adds approximately USD 23,651 to predicted 
  charges on average based on the model's learned coefficients
- Age had moderate influence at 0.30 correlation — each 
  additional year adds roughly USD 257 to charges
- BMI alone was a weak predictor at 0.20 correlation, but when 
  combined with smoking status the scatter plot revealed that 
  high BMI dramatically increases charges for smokers while 
  having little effect on non-smokers
- Sex, number of children, and region had minimal impact
- MAE of USD 4,181 means predictions are off by that amount 
  on average
- RMSE of USD 5,796 is higher than MAE, confirming that a 
  subset of predictions carry large errors — the model 
  struggles most with high-charge customers
- The residual plot showed a downward pattern rather than a 
  random cloud, indicating Linear Regression cannot fully 
  capture the non-linear relationship between smoking and charges

### Tools and Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
