# Processed Data Directory (`data/processed/`)

This directory contains intermediate processed datasets, metadata caches, and precomputed sample indexes.

## Rules:
- Preprocessing statistics (such as feature scaling or imputations) are calculated strictly on the training set and applied to validation/test sets to prevent data leakage.
- Cached files are excluded from Git version control.
