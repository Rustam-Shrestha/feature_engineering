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
