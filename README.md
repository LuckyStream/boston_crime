# Boston Crime Analysis

DS 4420 Final Project — Northeastern University

## Overview
We use three machine learning methods to analyze crime patterns in Boston:
- Time Series (SARIMA) to forecast monthly crime counts
- Neural Network to use MLP to classify shooting incidents from temporal and spatial features
- Bayesian Regression to model crime counts by district (coming soon)

## Data
Downloaded from [Analyze Boston](https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system). Place the yearly CSV files in a `data/` folder one level above this repo.

## Project Structure
```
Project/
├── data/               # not included in repo
└── boston_crime/       # this repo
    ├── data_prep.ipynb
    └── time_series.ipynb
```

## How to Run
1. Download crime CSVs from the link above and place in `../data/`
2. Run `data_prep.ipynb` to merge the data
3. Run `time_series.ipynb` for the SARIMA model

## Requirements
Python 3.8+, pandas, numpy, matplotlib, statsmodels

```python

```
