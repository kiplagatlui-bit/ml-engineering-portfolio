# Machine Learning Engineering Portfolio

**Portfolio owner:** [Your full name]  
**Location:** Kenya  
**Focus:** Python · ML pipelines · model evaluation · data validation · inference

> This repository is a practical, reproducible demonstration project. It is not a claim of production deployment or employment at an AI laboratory. Replace the profile placeholders and add only experience you can substantiate.

## Project: Reproducible classification pipeline

This project demonstrates an end-to-end tabular ML workflow using a generated dataset:

1. Generate a deterministic dataset.
2. Validate input features and labels.
3. Split data reproducibly into train and test sets.
4. Train a preprocessing + logistic-regression pipeline.
5. Evaluate with accuracy, precision, recall, F1, ROC-AUC and a confusion matrix.
6. Save the model and metrics as versionable artifacts.
7. Load the saved model for batch inference.
8. Run automated tests.

### Engineering practices demonstrated

- Reproducible random seeds and deterministic data generation.
- Separation of data, training, evaluation and inference code.
- Input schema and missing-value checks.
- Scikit-learn `Pipeline` to prevent preprocessing leakage.
- Machine-readable evaluation reports.
- Command-line interface and automated tests.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
```

Train and evaluate:

```bash
ml-portfolio train --output-dir artifacts
```

Run inference on a small example:

```bash
ml-portfolio predict --model artifacts/model.joblib --values 1.2,0.8,-0.4,2.1,0.5,1.0
```

Run tests:

```bash
pytest
```

## Repository structure

```text
src/ml_portfolio/
  data.py        deterministic dataset generation and validation
  modeling.py   pipeline construction, training and evaluation
  inference.py  model loading and prediction
  cli.py         command-line interface
tests/
  test_pipeline.py
docs/
  architecture.md
  project-notes.md
```

## Results

Run the training command to create `artifacts/metrics.json` and `artifacts/model.joblib`. Metrics are generated locally from the fixed synthetic dataset; do not copy results into a job application without running the project yourself and checking the output.

## Suggested extensions

- Add cross-validation and hyperparameter search.
- Compare a tree-based model against logistic regression.
- Add data-drift checks and model monitoring.
- Package the inference service with FastAPI.
- Add a CI workflow that runs tests on every pull request.
- Add experiment tracking and model version metadata.

## About the author

[Write 3–5 truthful sentences about your education, AI evaluation/annotation experience, Python experience and relevant projects. Do not list qualifications or employment that you cannot verify.]

## License

MIT. See `LICENSE`.
