# Task 7: Customer Segmentation Using Unsupervised Learning

## Objective

Segment mall customers into distinct groups based on their spending habits and personal attributes using unsupervised machine learning, and propose targeted marketing strategies for each identified segment.

## Dataset

**Mall Customers Dataset (Marketing Campaign)**

- 2,240 customer records and 29 columns
- Features include annual income, spending across product categories (wines, fruits, meat, fish, sweets, gold), marital status, education level, number of children, recency of last purchase, and campaign response history
- No target label — the goal is to discover natural groupings within the data

## Approach

- Loaded and inspected the dataset; handled 24 missing values in the `Income` column by imputing with the median
- Dropped irrelevant columns (`ID`, `Z_CostContact`, `Z_Revenue`)
- Engineered new features: `Age`, `TotalSpending`, `TotalPurchases`, `Family_Size`, `TotalCampaignsAccepted`, and `Customer_Days` (tenure)
- Encoded categorical features — education mapped to ordinal values, marital status simplified to a binary `Is_Partner` flag
- Removed outliers in `Income` (>600K) and `Age` (>100)
- Visualized distributions, spending by product category, income vs spending scatter, correlation heatmap, and demographic breakdowns
- Standardized features using `StandardScaler` before clustering (mandatory for distance-based models)
- Used the Elbow Method and Silhouette Score analysis to determine the optimal number of clusters (K=4)
- Trained K-Means with K=4 and evaluated using a Silhouette Plot
- Applied PCA (2D) and t-SNE for dimensionality reduction and visual confirmation of cluster separation
- Profiled each cluster using radar charts, box plots, and spending category breakdowns

## Results and Insights

- Income had the strongest correlation with Total Spending at **0.79**, making it the dominant feature driving customer separation
- Wines and Meat Products were the highest-spending categories by a large margin, concentrated in high-income clusters
- Graduation-level customers made up the largest demographic group; PhD and Master holders showed higher average spending
- K-Means with K=4 achieved a **Silhouette Score of 0.1294**; PCA and t-SNE both visually confirmed meaningful cluster separation

**Four customer segments identified:**

| Cluster | Label | Profile |
|---------|-------|---------|
| C0 | High-Value Premium Shoppers | High income, high spending, low family size, strong campaign responders |
| C1 | Budget-Conscious Families | Low-to-mid income, large families, low spending, minimal campaign response |
| C2 | Mid-Tier Active Buyers | Mid income, moderate spending, active across purchase channels |
| C3 | Low-Income Minimal Spenders | Lowest income, minimal spending, rarely engaged with campaigns |

## Business Recommendation

Cluster 0, despite being the smallest group, carries the highest revenue potential per customer and should be the primary focus for premium marketing campaigns — particularly around wines, meat, and gold products. Cluster 2 responds well to catalog and web channels and would benefit from loyalty offers. Clusters 1 and 3 are better targeted with budget-friendly deals and family-oriented promotions rather than premium upsells.

A more balanced dataset with additional behavioural signals such as browsing history or loyalty points would further improve segmentation quality.

## Tools and Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (KMeans, PCA, t-SNE, StandardScaler, Silhouette Score)
**Zainab**
Data Science & Analytics Intern – DevelopersHub Corporation
[GitHub](https://github.com/Zainab-71)
