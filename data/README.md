# Dataset Documentation — FreshScan AI

## 1. Dataset Identity & Provenance

- **Dataset Name**: Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality (`raw` un-augmented split)
- **Standardized Mirror**: [Project-AgML / `fresh_rotten_fruit_classification`](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification) (`config="raw"`)
- **Original Authors**: Nusrat Sultana, Musfika Jahan, Mohammad Shorif Uddin (2022)
- **Peer-Reviewed Data Article**: *An extensive dataset for successful recognition of fresh and rotten fruits*, Data in Brief, Vol. 44, 108552 (`https://doi.org/10.1016/j.dib.2022.108552`)
- **Primary Archive DOI**: `10.17632/bdd69gyhv8.1` (Mendeley Data, V1)
- **License**: **Creative Commons Attribution 4.0 International (`CC BY 4.0`)**

Why the `raw` configuration was chosen over pre-augmented mirrors:
Pre-augmented datasets frequently place rotated or zoomed copies of the same physical fruit photograph into both training and test folders, inflating test accuracy via data leakage. FreshScan AI downloads strictly the **un-augmented raw photographs** (`raw` config), performs exact-byte (`SHA-256`) and perceptual (`64-bit dHash`) deduplication, splits into stratified train/val/test partitions, and applies augmentation exclusively to training batches.

## 2. Supported Fruits & Class Labels (16 Fine-Grained Classes)

The dataset covers **8 fruit species**, each across **2 freshness states** (`fresh` and `rotten`):

| Fruit Species | Fresh Class Label | Rotten Class Label |
| :--- | :--- | :--- |
| Apple | `fresh_apple` | `rotten_apple` |
| Banana | `fresh_banana` | `rotten_banana` |
| Grape | `fresh_grape` | `rotten_grape` |
| Guava | `fresh_guava` | `rotten_guava` |
| Jujube | `fresh_jujube` | `rotten_jujube` |
| Orange | `fresh_orange` | `rotten_orange` |
| Pomegranate | `fresh_pomegranate` | `rotten_pomegranate` |
| Strawberry | `fresh_strawberry` | `rotten_strawberry` |

## 3. Directory Layout

```text
data/
├── README.md                  # This documentation file
├── raw/                       # Gitignored: Raw downloaded class folders (data/raw/<class_label>/*.jpg)
├── processed/                 # Gitignored: Deduplicated, stratified 224x224 uint8 split caches (*.npz)
└── sample_test_images/        # 16 real held-out test set samples (1 per class) + manifest.json for live UI testing
```
