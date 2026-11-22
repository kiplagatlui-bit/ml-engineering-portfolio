# Project notes and review checklist

Before sharing this repository:

- [ ] Install dependencies in a clean virtual environment.
- [ ] Run `ml-portfolio train --output-dir artifacts`.
- [ ] Run `pytest` and confirm all tests pass.
- [ ] Inspect `artifacts/metrics.json`; understand each metric.
- [ ] Run the inference command with six numeric values.
- [ ] Replace the author placeholder in `README.md` with accurate details.
- [ ] Add your own GitHub username and repository URL to your CV.
- [ ] Do not claim this is production experience; describe it as an independent portfolio project.

## Interview discussion prompts

1. Why is scaling placed inside a pipeline?
2. Why use a stratified split?
3. Why report precision and recall as well as accuracy?
4. What does ROC-AUC measure, and when can it be misleading?
5. How would you add cross-validation, drift detection and model monitoring?
6. What changes would be needed to serve predictions through an API?
