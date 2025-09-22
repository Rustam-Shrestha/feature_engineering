# Feature Engineering with Pandas

## Overview

This notebook explores key techniques in feature engineering using Pandas. It covers:

- Types of data
- Selecting specific data types
- Encoding categorical variables
- Handling uncommon categories
- Working with numeric variables
- Binarizing and binning
- Handling missing data

## Datasets Used

- Stack Overflow Developer Survey 2023  
  https://raw.githubusercontent.com/Stephen137/stack_overflow_developer_survey_2023/main/data/survey_results_public_2023.csv

- NYC Restaurant Inspection Results  
  https://data.cityofnewyork.us/api/views/43nn-pn8j/rows.csv?accessType=DOWNLOAD

## Topics Covered

### Types of Data

- Continuous  
- Categorical  
- Ordinal  
- Boolean  
- DateTime

### Data Type Selection

- Filtering numeric columns using `select_dtypes`

### Categorical Encoding

- One-Hot Encoding: Converts each category into a separate binary column
- Dummy Encoding: Similar to one-hot but drops one column to avoid collinearity

### Handling Rare Categories

- Grouping uncommon values under a single label such as "Other"

### Numeric Feature Engineering

- Binarizing violation scores from restaurant data
- Binning salary data into labeled ranges using `pd.cut`

### Missing Data Handling

- Checking for null values
- Inspecting dataset-wide missing data patterns


#Day 2
# Data Cleaning and Handling Missing Values with Pandas

## Overview

This notebook focuses on techniques for identifying and handling missing data, as well as cleaning numerical data using Pandas. It covers:

- Inspecting datasets for missing values
- Handling missing data with listwise deletion
- Replacing missing values with constants or statistical measures
- Cleaning numerical columns by removing stray characters
- Using method chaining for efficient data cleaning

## Datasets Used

- Stack Overflow Developer Survey 2023  
  https://raw.githubusercontent.com/Stephen137/stack_overflow_developer_survey_2023/main/data/survey_results_public_2023.csv

## Topics Covered

### Inspecting Missing Data

- Using `data.info()` to view dataset structure and data types
- Using `isnull()` to identify missing values
- Checking non-missing value counts with `count()`
- Displaying locations of missing and non-missing values

### Handling Missing Data

- **Listwise Deletion**: Dropping rows or columns with any missing values using `dropna()`
- **Subset Deletion**: Dropping rows with missing values in specific columns (e.g., `Gender`)
- **Replacing Missing Values**: 
  - Filling categorical missing values with a constant (e.g., 'Not Given')
  - Filling numerical missing values with the mean
  - Rounding numerical replacements for consistency
- Considerations for predictive modeling: Using train set statistics (mean/median) to fill missing values in both train and test sets to avoid data leakage

### Cleaning Numerical Data

- Removing stray characters (e.g., commas, dollar signs, pound signs) from numerical columns
- Converting object-type columns to numeric using `pd.to_numeric()` with error coercion
- Handling non-numeric values by coercing to `NaN`
- Using method chaining to streamline cleaning operations

### Key Considerations

- Trade-offs of listwise deletion: Loss of valid data and reduced model information
- Risks of filling with mean/median: Potential bias in variance and covariance estimates
- Importance of cleaning numerical data for machine learning compatibility
