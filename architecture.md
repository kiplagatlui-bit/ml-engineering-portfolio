# Architecture

```text
Synthetic data generator
        |
        v
Feature validation
        |
        v
Stratified train/test split
        |
        v
Pipeline: StandardScaler -> LogisticRegression
        |
        +------> Evaluation metrics (held-out test set)
        |                    |
        v                    v
 model.joblib          metrics.json
        |
        v
Batch inference CLI
```

## Design decisions

- **Pipeline encapsulation:** scaling is fitted only on training data because it is part of the scikit-learn pipeline. This avoids leaking test-set statistics into training.
- **Stratification:** preserves class proportions across train and test splits.
- **Fixed seed:** makes the demonstration repeatable.
- **Validation at boundaries:** feature matrices are checked before training and prediction.
- **Separation of concerns:** data generation, modeling, inference and CLI code are kept in separate modules.

## Limitations

This is a small synthetic-data demonstration, not a production service. It does not establish performance on a real business dataset, model fairness, security, drift monitoring, GPU optimization or production-scale serving.
