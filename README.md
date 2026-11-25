# FreshScan AI — Fruit Freshness Detection

A clean, beginner/intermediate **Python & Machine Learning portfolio project** built with **Pandas**, **NumPy**, **Matplotlib**, **Seaborn**, **Scikit-learn**, **Streamlit**, and an optional introductory **TensorFlow/Keras** learning lab.

---

## 1. Project Overview

FreshScan AI classifies whether a fruit sample is **Fresh** or **Spoiled** based on storage conditions (`storage_days`, `temperature_c`, `humidity_pct`) and physical quality measurements (`firmness_score`, `browning_pct`, `moisture_loss_pct`, `fruit_type`).

The project uses a clearly labeled **educational sample dataset** (`data/fruit_freshness_dataset.csv`) containing realistic missing values (`NaN`) and duplicate rows so every step of a real Data Science workflow works end-to-end:

1. **Load Dataset** (`pd.read_csv`)
2. **Explore Data** (`.head()`, `.shape`, `.dtypes`, `.describe()`)
3. **Check Missing Values & Duplicates** (`.isnull().sum()`, `.duplicated().sum()`)
4. **Clean & Preprocess Data** (`drop_duplicates`, median imputation, scaling, one-hot encoding)
5. **Visualize Data** (Bar chart, Pie chart, Line chart, Histogram, Correlation Heatmap)
6. **Split Data** (`train_test_split` with stratification)
7. **Train Simple Models** (`Logistic Regression`, `Decision Tree`, `Random Forest`)
8. **Use a Scikit-learn `Pipeline`** (`ColumnTransformer` + `SimpleImputer` + `StandardScaler` + `OneHotEncoder`)
9. **Evaluate Results** (`accuracy_score`, `confusion_matrix`, `classification_report`)
10. **Make Live Predictions & Optional Intro TensorFlow Demo**

---

## 2. Project Structure

```text
FreshScan-AI/
├── app.py                            # Streamlit UI (Overview, Data Analysis, Visualizations, ML, Prediction, TF Lab)
├── ml_pipeline.py                    # Clean helper functions for loading, cleaning, Scikit-learn Pipeline & TensorFlow demo
├── data/
│   └── fruit_freshness_dataset.csv   # Educational sample CSV dataset (with realistic missing values & duplicates)
├── tests/
│   └── test_pipeline.py              # Pytest checks for cleaning, training, evaluation, and predictions
├── requirements.txt                  # Pure Python libraries (Streamlit, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow)
└── README.md                         # Project documentation
```

---

## 3. Run Locally

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit app
streamlit run app.py
```

### Run Tests

```bash
pytest -v
```

---

## 4. Tech Stack (Interview Talking Points)

- **Pandas & NumPy**: CSV loading, inspecting missing data, summary statistics, and numerical arrays.
- **Matplotlib & Seaborn**: Bar chart, line chart, pie chart, histogram, correlation heatmap, and confusion matrix heatmap.
- **Scikit-learn**: `train_test_split`, `ColumnTransformer`, `Pipeline`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `accuracy_score`, and `confusion_matrix`.
- **Streamlit**: Clean interactive Python web UI without needing a separate backend API server.
- **TensorFlow / Keras (Optional Section)**: Small 3-layer `Sequential` model demonstrating Input Layer, Hidden Layers, Output Layer, Epochs, Loss, and Accuracy.
