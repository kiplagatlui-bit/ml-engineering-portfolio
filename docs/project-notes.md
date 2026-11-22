# Project Notes and Review Checklist

## Setup and Testing

- Install dependencies in a clean Python virtual environment.
- Run ml-portfolio train --output-dir artifacts.
- Run pytest and confirm all tests pass.
- Inspect artifacts/metrics.json.
- Run the inference command with six numeric values.

## Interview Preparation

1. Why is scaling placed inside a machine-learning pipeline?
2. Why use a stratified train/test split?
3. Why report precision and recall alongside accuracy?
4. What does ROC-AUC measure?
5. How could cross-validation and model monitoring be added?
6. How could predictions be served through an API?

## Project Limitations

This is an independent portfolio project using synthetic data. It demonstrates fundamental ML engineering practices, not production deployment experience.

## Next Improvements

- Add cross-validation.
- Add experiment tracking.
- Build a prediction API.
- Add model monitoring.
- Explore Docker-based deployment.
