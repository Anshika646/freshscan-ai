# FreshScan AI — Fruit Freshness Detection

An end-to-end computer vision product for optical produce freshness classification built with **TensorFlow / Keras (`MobileNetV2` transfer learning)**, **FastAPI**, **React 19 + TypeScript**, **Three.js procedural 3D WebGL**, **GSAP + ScrollTrigger**, and **Express** — engineered for reproducible evaluation, GitHub (`FreshScan-AI`), Docker, and Cloud Run deployment.

> **Food Safety Disclaimer**: Visual classification detects external peel discoloration, bruising, and surface mold only. Optical predictions cannot detect internal core rot, bacterial pathogens (*Salmonella*, *E. coli*, *Listeria*), mycotoxins, or chemical residue. Never rely solely on an image classifier to guarantee food safety.

---

## 1. System Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                     Browser Client (React 19 + TypeScript)                   │
│  ┌────────────────────────────┐  ┌─────────────────────────────────────────┐ │
│  │ Three.js Full-Viewport 3D  │  │ GSAP + ScrollTrigger Foreground UI      │ │
│  │ 7 Procedural Fruit Models  │◄─┤ Hero, Live Optical Scanner, 16x16       │ │
│  │ + Optical Gimbal + Scroll  │  │ Confusion Matrix, Model & Deploy Panels │ │
│  └────────────────────────────┘  └───────────────────┬─────────────────────┘ │
└──────────────────────────────────────────────────────┼───────────────────────┘
                                                       │ HTTP / Multipart
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 Express Full-Stack Gateway (`server.ts`, :3000)              │
│   • Serves built Vite SPA (`dist/`) or Vite dev middleware                   │
│   • Proxies `/health`, `/model-info`, `/metrics`, `/predict`, `/api/ml/*`    │
│   • Supervises Python worker + `/api/deploy/readiness` + GitHub OAuth sync   │
└──────────────────────────────────────────────────────┬───────────────────────┘
                                                       │ Local Proxy (:8050)
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│               FastAPI Inference & Analytics Backend (`app.py`)               │
│   • Validates extension, MIME, byte size (`MAX_UPLOAD_MB`), & truncation     │
│   • `#FFFFFF` alpha compositing → `224x224` Bilinear → `[-1, 1]` scaling     │
│   • Real TensorFlow/Keras `MobileNetV2` (`models/freshscan_mobilenetv2.keras`)│
│   • Serves verified test metrics, plots, & 16 unseen held-out test specimens │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Verified Held-Out Test Performance

Evaluated strictly on **`112` unseen held-out test images** (`seed=42`, zero `SHA-256` byte overlap and zero 64-bit `dHash` perceptual overlap across splits):

| Metric | Measured Result |
| --- | --- |
| **Held-Out Test Accuracy (16 Classes)** | **`97.32%`** (`109 / 112`) |
| **Macro F1-Score** | **`97.31%`** |
| **Macro Precision / Macro Recall** | **`97.66%`** / **`97.32%`** |
| **Multi-Class OvR ROC-AUC** | **`0.9995`** |
| **Binary Fresh vs. Rotten Accuracy** | **`97.32%`** |
| **Saved Keras v3 Artifact Size** | **`8.92 MB`** (`freshscan_mobilenetv2.keras`) |

### Model Progression Comparison

| Stage | Description | Accuracy |
| --- | --- | --- |
| Uniform Random Baseline | 16 balanced classes (`1 / 16`) | `6.25%` |
| Stage 1 Epoch 1 (Frozen Backbone) | Transfer head warm-up before convergence | `77.68%` (val) |
| Stage 2 Fine-Tuned `MobileNetV2` | Top 30 backbone layers unfrozen (`lr=1e-4`, cosine decay) | **`97.32%`** (held-out test) |

---

## 3. Dataset Provenance & License

- **Dataset**: *Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality* (un-augmented `raw` split)
- **Standardized Mirror**: [Hugging Face — `Project-AgML/fresh_rotten_fruit_classification`](https://huggingface.co/datasets/Project-AgML/fresh_rotten_fruit_classification) (`config="raw"`)
- **Authors**: Nusrat Sultana, Musfika Jahan, Mohammad Shorif Uddin (2022)
- **Publication**: *Data in Brief*, Vol. 44, 108552 (`https://doi.org/10.1016/j.dib.2022.108552`), Mendeley Data V1 (`doi:10.17632/bdd69gyhv8.1`)
- **License**: **Creative Commons Attribution 4.0 International (`CC BY 4.0`)**
- **Supported Produce (8) & Class Labels (16)**:
  - `fresh_apple`, `rotten_apple`
  - `fresh_banana`, `rotten_banana`
  - `fresh_grape`, `rotten_grape`
  - `fresh_guava`, `rotten_guava`
  - `fresh_jujube`, `rotten_jujube`
  - `fresh_orange`, `rotten_orange`
  - `fresh_pomegranate`, `rotten_pomegranate`
  - `fresh_strawberry`, `rotten_strawberry`

---

## 4. Data Preprocessing & Leakage Prevention

1. **Strict Image Validation**: Rejects empty files, unsupported extensions/MIME types, truncated streams (`ImageFile.LOAD_TRUNCATED_IMAGES = False`), oversized uploads (`HTTP 413`), and sub-`16×16` thumbnails.
2. **White-Background Alpha Compositing**: RGBA/transparent PNGs are composited onto `#FFFFFF` prior to 3-channel `RGB` conversion so transparent alpha regions never become `0,0,0` black pixels that mimic necrotic rot.
3. **Exact & Perceptual Deduplication Before Splitting**:
   - Exact byte duplicates (`SHA-256`) are removed prior to partitioning.
   - Near-duplicate / re-compressed images sharing an identical 64-bit difference hash (`dHash`) are grouped into atomic clusters assigned together to a single split (`train`, `val`, or `test`).
4. **Stratified `70% / 15% / 15%` Partitioning**: Preserves class distribution across training (`504`), validation (`104`), and held-out test (`112`) splits (`seed=42`).
5. **Color-Preserving Training Augmentation**: Horizontal flips, `±8%` rotation, `±10%` zoom, and `±10%` contrast jitter are applied **exclusively to training batches**. Hue shifts are excluded because surface hue is biologically diagnostic of oxidation and mold.
6. **Balanced Class Weights**: Computed strictly from training split frequencies (`compute_class_weight('balanced')`).

---

## 5. REST API Reference

Both standalone FastAPI (`app.py` on `:8050`) and the full-stack Express gateway (`server.ts` on `:3000`) expose:

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` (alias `/api/ml/health`) | Service status, TensorFlow version, model readiness, and upload limits |
| `GET` | `/model-info` (alias `/api/ml/model-info`) | Architecture, input shape, 16 classes, preprocessing spec, and provenance |
| `GET` | `/metrics` (alias `/api/ml/metrics`) | Held-out test metrics, confusion matrix, ROC-AUC, and baseline comparison |
| `POST` | `/predict` (alias `/api/ml/predict`) | Multipart `file` upload returning real `MobileNetV2` softmax probabilities |
| `GET` | `/api/ml/analytics` | Bundled dataset summary, epoch training history, and evaluation report |
| `GET` | `/api/ml/samples` | Lists 16 unseen held-out test specimens for instant live verification |
| `POST` | `/api/ml/predict-sample/:sampleId` | Runs live TensorFlow inference on a held-out specimen & compares ground truth |
| `GET` | `/api/deploy/readiness` | Live 8-check production readiness audit across model, metrics, and SPA bundle |

---

## 6. Repository Structure

```text
FreshScan-AI/
├── .github/workflows/ci.yml         # Automated pytest + TypeScript lint + Vite build CI
├── LICENSE                          # MIT code license + CC BY 4.0 dataset attribution
├── CONTRIBUTING.md                  # Leak-free ML contribution & testing workflow
├── README.md                        # Architecture, metrics, API, and deployment guide
├── Dockerfile                       # Multi-runtime (Node 22 + Python TensorFlow CPU) container
├── requirements.txt                 # Python ML, FastAPI, Pillow, scikit-learn, pytest deps
├── package.json                     # React 19, Three.js, GSAP, Tailwind v4, Express scripts
├── train_model.py                   # Reproducible 2-stage MobileNetV2 training CLI
├── evaluate_model.py                # Held-out test evaluation & Matplotlib PNG chart exporter
├── app.py                           # FastAPI inference & metrics server
├── server.ts                        # Express full-stack gateway & process supervisor
├── data/
│   ├── README.md                    # Dataset license, citation, and download details
│   └── sample_test_images/          # 16 unseen held-out test specimens + manifest.json
├── models/
│   ├── freshscan_mobilenetv2.keras  # Trained Keras v3 MobileNetV2 classifier (8.92 MB)
│   ├── class_indices.json           # Deterministic 16-class label mapping
│   ├── preprocessing_config.json    # Input dimensions, RGB compositing & scaling config
│   ├── dataset_summary.json         # Raw dataset exploration, dimensions & duplicate audit
│   ├── training_history.json        # Epoch-by-epoch Stage 1 & Stage 2 training curves
│   ├── evaluation_report.json       # Held-out test metrics, confusion matrix & limitations
│   └── plots/                       # Generated PNG charts (confusion_matrix, training_curves, etc.)
├── src/
│   ├── preprocessing.py             # Image validation, RGB compositing, leak-free split logic
│   ├── data_loader.py               # Hugging Face raw dataset downloader & split builder
│   ├── predict.py                   # Stateful TensorFlow inference + uncertainty flags
│   ├── three/                       # Procedural 3D fruit models + scroll-synced WebGL stage
│   ├── components/                  # Dark UI views (GSAP Overview, Scanner, Analytics, Deploy)
│   └── App.tsx                      # Root application state & 3D background integration
└── tests/
    └── test_ml_pipeline.py          # Pytest unit, API, error-handling & production model suite
```

---

## 7. Local Setup, Retraining & Testing

### Install Dependencies

```bash
python3 -m venv --copies .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
npm install
```

### Retrain & Re-Evaluate From Raw Benchmark Images

```bash
# 1. Download raw benchmark photos, audit dimensions/duplicates, and build leak-free splits
.venv/bin/python -m src.data_loader --max-per-class 45 --cleanup-raw --seed 42

# 2. Train 2-stage MobileNetV2 transfer learning classifier
.venv/bin/python train_model.py --no-download --stage1-epochs 8 --stage2-epochs 6 --batch-size 32 --seed 42

# 3. Evaluate on unseen held-out test split & export diagnostic PNG figures
.venv/bin/python evaluate_model.py
```

### Run Verification Suite

```bash
.venv/bin/pytest -v
npm run lint
npm run build
```

### Start the Full-Stack Application

```bash
# Development mode (Express on :3000 + FastAPI TensorFlow worker on :8050)
npm run dev

# Production mode (serves compiled dist/ + FastAPI TensorFlow worker)
NODE_ENV=production npm start
```

---

## 8. GitHub Sync (`FreshScan-AI`) & Production Deployment

### Push to GitHub (`FreshScan-AI`)

```bash
git init -b main
git add .
git commit -m "FreshScan AI: end-to-end MobileNetV2 freshness detection with 3D WebGL & GSAP UI"
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/FreshScan-AI.git
git push -u origin main
```

### Option A — Single-Container Unified Deployment (Google Cloud Run / Docker)

TensorFlow CPU runtime requires **`>= 1.5 GiB`–`2 GiB` RAM** at startup (standard `512 MB` hobby containers will OOM when loading `libtensorflow`).

```bash
gcloud run deploy freshscan-ai \
  --source . \
  --region asia-east1 \
  --memory 2Gi \
  --cpu 2 \
  --port 8080 \
  --allow-unauthenticated
```

Or run anywhere via Docker:

```bash
docker build -t freshscan-ai .
docker run --rm -p 8080:8080 freshscan-ai
```

### Option B — Zero-Cost Split Deployment (Hugging Face Spaces + Vercel / Netlify)

1. **ML Backend (`16 GB` RAM Free Tier)**: Deploy `Dockerfile` or `app.py` + `models/` to a **Hugging Face Docker Space** and set `CORS_ORIGINS=https://your-frontend.vercel.app`.
2. **Static Frontend**: Deploy `dist/` (`npm run build`) to Vercel, Netlify, or Cloudflare Pages pointing `/api/ml/*` rewrites to your Space URL.

---

## 9. Known Limitations

- Evaluates **single-item studio/countertop framing** best; cluttered multi-fruit scenes without bounding-box cropping degrade softmax calibration.
- Supports the **8 documented produce types** (`Apple`, `Banana`, `Grape`, `Guava`, `Jujube`, `Orange`, `Pomegranate`, `Strawberry`); out-of-distribution produce triggers low-confidence / narrow-margin warnings.
- Surface RGB imaging cannot inspect internal core rot or microbial contamination.
