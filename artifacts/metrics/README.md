# Evaluation Metrics (`artifacts/metrics/`)

Stores quantitative benchmark results on the isolated holdout test set:
1. `evaluation_summary.json`: Detailed classification report for Custom CNN, EfficientNetB0, ResNet50, and Stacking Meta-Classifier (Accuracy, Balanced Accuracy, Macro F1, Weighted F1, Per-Class Precision & Recall).
2. `confusion_matrices.json`: Raw and normalized confusion matrices across all 7 diagnostic classes.
3. `roc_curves_data.json`: True positive rate and false positive rate coordinates for ROC-AUC analysis.
