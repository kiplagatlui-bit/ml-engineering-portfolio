# Architecture

## Machine Learning Workflow

Synthetic data generation
→ Feature validation
→ Stratified train/test split
→ StandardScaler preprocessing
→ Logistic Regression training
→ Model evaluation
→ Model and metrics storage
→ Batch inference

## Design Decisions

- Pipeline encapsulation: scaling is fitted on training data only.
- Stratification: preserves class proportions in training and testing.
- Fixed random seed: makes experiments reproducible.
- Feature validation: checks input data before training and prediction.
- Modular design: separates data generation, modeling, inference and CLI operations.

## Evaluation Metrics

The project reports:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Limitations

This is an independent demonstration using synthetic data. It is not a production ML system and has not been validated on real-world business data.
