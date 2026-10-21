# Week 1 task

This folder contains the first Python data-processing assignment.

## Files

- `students.csv` — the raw dataset with intentional data-quality issues.
- `task.py` — contains the cleaning and analysis workflow.
- `task.ipynb` — a notebook version of the same project for Jupyter and Colab.
- `README.md` — quick instructions for this week.

## How to run it

From the repository root:

```bash
python week-01/task.py
```

## What the project does

- loads raw CSV data
- standardizes text values
- converts values to correct data types
- handles missing values using median imputation
- removes duplicates
- checks for invalid numeric values
- calculates statistics with NumPy

This project is designed to be easy to understand for a beginner while still showing realistic data-cleaning practice.
