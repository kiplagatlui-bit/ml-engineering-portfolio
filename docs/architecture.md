# Architecture

## Machine Learning Workflow

Synthetic data generation → Feature validation → Stratified train/test split → StandardScaler → Logistic Regression → Model evaluation → Model storage → Batch inference.

## Design Decisions

- Scaling is fitted only on training data.
- Stratification preserves class proportions.
- A fixed random seed supports reproducibility.
- Feature validation checks inputs before training and prediction.
- Separate modules handle data, modeling, inference and CLI operations.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## Limitations

This is an independent demonstration using synthetic data. It is not a production ML system and has not been validated on real-world business data.
