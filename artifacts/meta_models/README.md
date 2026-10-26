# Meta-Models (`artifacts/meta_models/`)

Stores artifacts related to the Stacking Ensemble Meta-Learner:
1. `oof_predictions.npy`: Out-of-fold probability matrix generated strictly via cross-validation without data leakage.
2. `stacking_meta_model.joblib`: Serialized scikit-learn `LogisticRegression` meta-classifier.
3. `meta_feature_names.json`: Schema mapping feature indices to `(model_name, class_name)` pairs.
