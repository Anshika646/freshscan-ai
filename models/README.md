# Model Artifacts — FreshScan AI

## 1. Architecture Summary

- **Backbone**: `tf.keras.applications.MobileNetV2` (`input_shape=(224, 224, 3)`, `include_top=False`, `weights="imagenet"`)
- **Classification Head**:
  1. `GlobalAveragePooling2D`
  2. `Dropout(0.25)`
  3. `Dense(192, activation="relu")`
  4. `Dropout(0.20)`
  5. `Dense(16, activation="softmax")`
- **Training Strategy**:
  - **Stage 1 (Head Warm-up)**: Base `MobileNetV2` frozen (`training=False` to preserve ImageNet `BatchNormalization` moving statistics), Adam (`lr=1e-3`), balanced training class weights, `EarlyStopping` + `ReduceLROnPlateau`.
  - **Stage 2 (Fine-Tuning)**: Top 30 `MobileNetV2` layers unfrozen (with `BatchNormalization` layers kept frozen), Adam (`lr=1e-4`), `EarlyStopping` restoring best validation weights.

## 2. Generated Artifacts

| File | Tracked in Git | Purpose |
| :--- | :--- | :--- |
| `freshscan_mobilenetv2.keras` | Yes (~9–10 MB, stripped of optimizer state) | Inference-ready Keras v3 model used by `src/predict.py` and `app.py` |
| `class_indices.json` | Yes | Deterministic `class_to_index` and `index_to_class` mapping |
| `preprocessing_config.json` | Yes | Exact input resolution (`224x224`), RGB alpha compositing, and normalization spec |
| `dataset_summary.json` | Yes | Raw dataset inspection metrics, dimensions, corrupted count, and duplicate/split audit |
| `training_history.json` | Yes | Epoch-by-epoch training & validation loss/accuracy across Stage 1 and Stage 2 |
| `evaluation_report.json` | Yes | Held-out test accuracy, per-class Precision/Recall/F1, confusion matrix, ROC-AUC, and limitations |
| `plots/*.png` | Yes | Diagnostic charts (`confusion_matrix.png`, `training_curves.png`, `class_distribution.png`) |

## 3. Retraining & Re-Evaluating

```bash
python train_model.py --max-per-class 60 --stage1-epochs 7 --stage2-epochs 6 --batch-size 32 --seed 42
python evaluate_model.py
```
