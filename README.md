# Machine Learning Engineering Portfolio

*Portfolio owner:* Lui Kiplagat Mibey  
*Location:* Kenya  
*Focus:* Python · Machine Learning · Model Evaluation · Data Validation · AI Training

## Project: Reproducible Machine Learning Pipeline

This project demonstrates an end-to-end machine learning workflow using Python and Scikit-learn.

The pipeline includes:

1. Deterministic synthetic dataset generation.
2. Input feature validation.
3. Reproducible train/test splitting.
4. Data preprocessing using StandardScaler.
5. Classification using LogisticRegression.
6. Model evaluation using multiple performance metrics.
7. Model and metrics persistence.
8. Batch inference through a command-line interface.
9. Automated testing.

## Technologies

- Python
- Scikit-learn
- NumPy
- Pandas
- Joblib
- Pytest
- Ruff
- Git and GitHub

## Engineering Practices

- Reproducible experiments using fixed random seeds.
- Separation of data generation, training, evaluation and inference.
- Input validation and missing-value checks.
- Scikit-learn pipelines to prevent preprocessing leakage.
- Automated testing and code-quality checks.
- Model persistence and batch prediction.

## Quick Start

Requires Python 3.10 or newer.

Create and activate a virtual environment:

bash
py -m venv .venv
.venv\Scripts\activate


Install dependencies:

bash
py -m pip install -e ".[dev]"


Train the model:

bash
ml-portfolio train --output-dir artifacts


Run batch inference:

bash
ml-portfolio predict --model artifacts/model.joblib --values 0.1,0.2,0.3,0.4,0.5,0.6


Run automated tests:

bash
pytest


Run code-quality checks:

bash
ruff check .


## Model Evaluation Results

The current synthetic dataset produced the following evaluation results:

| Metric | Result |
|---|---:|
| Accuracy | 82.08% |
| Precision | 87.01% |
| Recall | 67.00% |
| F1 Score | 75.71% |
| ROC-AUC | 85.78% |

*Test status:* 4 automated tests passed.

These results come from a synthetic dataset and should not be interpreted as evidence of real-world model performance.

## Repository Structure

text
ml-engineering-portfolio/
├── docs/
│   ├── architecture.md
│   └── project-notes.md
├── src/
│   └── ml_portfolio/
│       ├── __init__.py
│       ├── cli.py
│       ├── data.py
│       ├── inference.py
│       └── modeling.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── LICENSE
├── README.md
└── pyproject.toml


## About the Author

I am Lui Kiplagat Mibey, based in Kenya, with an interest in artificial intelligence, machine learning engineering, and data quality. I am developing practical skills in Python, machine learning workflows, model evaluation, and AI training. This portfolio showcases my independent technical projects and commitment to building reliable, reproducible machine learning solutions.

## Future Improvements

- Cross-validation and hyperparameter tuning.
- Comparison of multiple classification algorithms.
- Data drift detection and model monitoring.
- API-based model deployment using FastAPI.
- Automated continuous integration with GitHub Actions.
- Experiment tracking and model versioning.

## License

This project is licensed under the MIT License.