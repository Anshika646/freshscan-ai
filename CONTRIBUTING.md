# Contributing to FreshScan AI

Thank you for your interest in improving **FreshScan AI**. To maintain scientific integrity, leak-free evaluation, and deployment reliability, please follow the guidelines below.

## 1. Machine Learning Integrity Rules

1. **Zero Data Leakage**: Never merge training and test splits before running SHA-256 exact deduplication and 64-bit `dHash` perceptual cluster grouping (`src/preprocessing.py`).
2. **Held-Out Test Isolation**: Never tune learning rates, early-stopping patience, or augmentation parameters against `test_split.npz`. Use `val_split.npz` exclusively for model selection.
3. **Color-Preserving Augmentation**: Do not introduce random hue inversion or extreme channel shuffling to training augmentation—peel chromaticity is a primary biological indicator of oxidation and fungal spoilage.
4. **Honest Metrics**: Never hardcode accuracy numbers in frontend components. All evaluation charts and KPIs must read directly from `models/evaluation_report.json` and `models/training_history.json`.

## 2. Local Development Workflow

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv --copies .venv
   .venv/bin/pip install -r requirements.txt
   npm install
   ```
2. Run the automated verification suite before opening a Pull Request:
   ```bash
   .venv/bin/pytest -v
   npm run lint
   npm run build
   ```

## 3. Repository Hygiene

- Never commit `.env` files, API keys, OAuth secrets, `.venv/`, `node_modules/`, raw dataset archives (`data/raw/`), or `.npz` tensor caches (`data/processed/`).
- Ensure all pull requests target the `FreshScan-AI` repository.
