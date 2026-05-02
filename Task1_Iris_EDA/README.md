# Task 1 — Exploring and Visualizing the Iris Dataset

## Task Objective
Understand how to read, summarize, and visualize a dataset using Python.
The goal was to explore the Iris dataset, inspect its structure, clean it,
and extract meaningful insights through visualizations.

## Approach
1. Loaded the dataset using Pandas
2. Inspected structure using .head(), .shape, .columns and .describe()
3. Checked for missing values and duplicate rows — removed 1 duplicate
4. Created histograms to examine distribution of all features
5. Created a scatter plot to analyze relationships between variables
6. Created a pair plot to visualize all feature combinations at once
7. Created box plots to detect outliers and spread across species

## Results & Insights
- Petal measurements vary far more than sepal measurements across species
- Sepal width is the weakest feature for distinguishing species (std = 0.44)
- Petal length is the strongest feature for distinguishing species (std = 1.77)
- Setosa is completely distinct from Versicolor and Virginica in all 
  petal measurements — the most easily identifiable species
- A strong positive correlation exists between sepal length and petal length
- Versicolor had one outlier detected near 3.0 cm in petal length
- A machine learning model trained on petal features would likely achieve
  high classification accuracy
