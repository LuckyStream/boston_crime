# Boston Crime Analysis
DS 4420 Final Project — Northeastern University

## Overview
We use three machine learning methods to analyze crime patterns in Boston:
- **Time Series (SARIMA)** to forecast monthly crime counts
- **MLP Neural Network** to classify whether an incident involves a shooting
- **Bayesian Logistic Regression** to classify shooting incidents using MCMC sampling

## Data
Downloaded from [Analyze Boston](https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system). Place the yearly CSV files in a `data/` folder one level above this repo.

## Project Structure
```
Project/
├── data/                  # not included in repo
└── boston_crime/          # this repo
    ├── data_prep.ipynb    # data merging and cleaning
    ├── time_series.ipynb  # SARIMA model (Python)
    ├── mlp_shooting.ipynb # MLP classifier (Python, NumPy)
    ├── bayesian_model.Rmd # Bayesian logistic regression (R)
    ├── bayesian_model.pdf # Bayesian model output
    ├── report.pdf         # final project report
    └── poster.pdf         # virtual poster
```

## How to Run
1. Download crime CSVs from the link above and place in `../data/`
2. Run `data_prep.ipynb` to merge and clean the data
3. Run `time_series.ipynb` for the SARIMA model
4. Run `mlp_shooting.ipynb` for the MLP classifier
5. Knit `bayesian_model.Rmd` in RStudio for the Bayesian model

## Requirements
**Python:** 3.8+, pandas, numpy, matplotlib, statsmodels

**R:** ggplot2, dplyr
