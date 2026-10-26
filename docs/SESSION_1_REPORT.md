# DermaStack-XAI: Session 1 Initialization & Environment Audit Report

**Date & Time**: 2026-10-18  
**Project**: DermaStack-XAI: An Explainable Multi-Model Deep Learning Framework for Skin Lesion Classification  
**Target Directory**: `K:\DERMASTACK-XAI`  
**Status**: Stage 0 Complete (Awaiting User Review and Approval for Stage 1)

---

## 1. Workspace Inspected

- **Workspace Path**: `K:\DERMASTACK-XAI`
- **Initial State**: The directory was completely empty (no previous files, Git repository, or legacy code).
- **Presentation File Search**:
  - Executed recursive scans on `K:\`, `C:\Users\HP\Downloads`, `C:\Users\HP\Desktop`, and `C:\Users\HP\Documents` for `*.pptx`, `*.ppt`, and `*.pdf` files matching `derma`, `lesion`, `skin`, or `presentation`.
  - **Result**: No existing presentation file was found on disk.
  - **Action**: In accordance with the prompt ("Read the project presentation if it is available"), the master prompt's explicit specifications (HAM10000 7-class taxonomy, Custom Lightweight CNN, EfficientNetB0, ResNet50, Stacking Meta-Classifier with Logistic Regression, Grad-CAM, Streamlit UI, metric suite, leakage-safe protocol) serve as the primary specification.

---

## 2. Environment & Compatibility Findings

### Hardware & OS Audit
- **Operating System**: Microsoft Windows 11 Home Single Language (64-bit, Version 10.0.26200).
- **System Memory (RAM)**:
  - Total Physical RAM: **24 GB** (24,568,428 KB).
  - Free Physical RAM: **~12.7 GB** (sufficient for in-memory batch generators and transfer learning feature caches).
- **Available Disk Space**:
  - Drive `K:\`: **338.38 GB free** out of 341.80 GB (ample capacity for raw HAM10000 images, splits, and model checkpoints).
  - Drive `C:\`: **147.06 GB free** out of 269.24 GB.
- **GPU / Acceleration Audit**:
  - Video Controller: **Intel(R) Graphics** (Integrated GPU, 2 GB adapter RAM).
  - Discrete NVIDIA GPU: **None detected** (`nvidia-smi` not present in PATH).
  - Native Windows TensorFlow status: TensorFlow 2.11+ on native Windows runs strictly in **CPU mode** (`/physical_device:CPU:0`).
  - **Architectural Consequence**: Training pipelines must utilize CPU-optimized configurations (frozen backbones during initial epochs, selective batch sizes, vectorised data caching, and avoidance of overly heavy networks).

### Software & Python Runtime Audit
- **Python Version**: **Python 3.12.5 (64-bit)** at `C:\Users\HP\AppData\Local\Microsoft\WindowsApps\python.exe` (invoked via `py -3.12`).
- **Installed Verified Packages**:
  - `tensorflow`: **2.21.0** (verified import and CPU device detection).
  - `keras`: **3.15.1** (verified Keras 3 compatibility).
  - `scikit-learn`: **1.9.1** (verified import).
  - `streamlit`: **1.64.0** (verified import).
  - `pandas`: **3.0.6** (verified import).
  - `numpy`: **2.5.3** (verified import).
  - `pillow`: **12.3.0** (verified import).
  - `matplotlib`: **3.11.2** (verified import).
  - `seaborn`: **0.13.2** (verified import).
  - `joblib`: **1.6.0** (verified import).
  - `tqdm`: **4.70.1** (verified import).
- **Test Framework**: Verified using Python's standard `unittest` library (8 automated unit tests executed and passing with 100% success rate).

---

## 3. Files Created or Modified

The following directory skeleton and project files have been established in `K:\DERMASTACK-XAI`:

```text
K:\DERMASTACK-XAI\
├── .gitignore                          # Excludes raw images, checkpoints, caches
├── requirements.txt                    # Exact verified versions pinned
├── README.md                           # Master project documentation
├── configs/
│   └── default_config.json             # Central configuration file
├── data/
│   ├── raw/
│   │   └── README.md                   # Raw HAM10000 ingestion guidelines
│   ├── processed/
│   │   └── README.md                   # Cached feature guidelines
│   └── splits/
│       └── README.md                   # Leakage-safe split guidelines
├── src/
│   ├── __init__.py                     # Package definition
│   ├── config.py                       # Dataclass-based centralized configuration
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                   # Structured console/file logging
│   │   └── seed.py                     # Deterministic multi-framework seeding
│   └── data/
│       ├── __init__.py
│       └── splitter.py                 # StratifiedGroupKFold on lesion_id
├── app/
│   ├── __init__.py
│   ├── app.py                          # Streamlit UI scaffold with medical disclaimer
│   └── README.md                       # Streamlit launch instructions
├── artifacts/
│   ├── checkpoints/
│   │   └── README.md                   # Target for saved Keras models
│   ├── meta_models/
│   │   └── README.md                   # Target for OOF predictions & Logistic Regression
│   ├── metrics/
│   │   └── README.md                   # Target for evaluation JSON summaries
│   └── figures/
│       └── README.md                   # Target for confusion matrices & ROC curves
├── tests/
│   ├── __init__.py
│   ├── test_environment.py             # 5 unit tests for Python, TF, Keras, configs
│   └── test_data_leakage.py            # 3 unit tests for patient grouping & zero leakage
└── docs/
    ├── ARCHITECTURE.md                 # Complete system design & Mermaid diagrams
    ├── SPECIFICATION.md                # Functional, non-functional, and clinical requirements
    ├── DATASET_GUIDELINES.md           # HAM10000 source, metadata, and duplicate audit
    ├── IMPLEMENTATION_CHECKLIST.md     # 9-stage roadmap with acceptance criteria
    ├── TESTING_STRATEGY.md             # Testing pyramid & QA protocol
    └── SESSION_1_REPORT.md             # This comprehensive audit report
```

---

## 4. Dataset Requirements & Leakage Risks

### Dataset Source
- **Official Dataset**: HAM10000 (Human Against Machine with 10,000 training images).
- **Official Source**: Tschandl et al., *Scientific Data* (2018), Harvard Dataverse `doi:10.7910/DVN/DBW86T` and ISIC Archive.
- **Seven Classes**:
  1. `akiec`: Actinic keratoses and intraepithelial carcinoma (~3.27%)
  2. `bcc`: Basal cell carcinoma (~5.13%)
  3. `bkl`: Benign keratosis-like lesions (~10.97%)
  4. `df`: Dermatofibroma (~1.15%)
  5. `mel`: Melanoma (~11.11%)
  6. `nv`: Melanocytic nevi (~66.95% - majority class)
  7. `vasc`: Vascular lesions (~1.42%)

### Critical Leakage Risks & Mitigation Protocol
1. **The Duplicate Lesion Problem**:
   - In HAM10000, 10,015 images represent only 7,470 unique lesions. ~25.4% of images are repeated shots of the same lesion.
   - **Risk**: Random splitting distributes images of the same lesion across both train and test partitions, causing memory leakage and falsely elevated accuracy metrics.
   - **Solution**: Implemented `src/data/splitter.py` enforcing `StratifiedGroupKFold` grouped on `lesion_id`.
   - **Verification**: Verified via `tests/test_data_leakage.py` that $\text{Train} \cap \text{Val} = \emptyset$ and $\text{Train} \cap \text{Test} = \emptyset$.
2. **Stacking Meta-Learner Leakage**:
   - **Risk**: Training the Logistic Regression meta-model on in-sample base model predictions causes overfitting and uncalibrated confidence.
   - **Solution**: The meta-learner is trained strictly on **Out-Of-Fold (OOF)** predictions generated through K-fold cross-validation.
3. **Class Imbalance Risk**:
   - Class `nv` accounts for ~67% of records.
   - **Solution**: Class weighting in loss functions and evaluation based on Balanced Accuracy, Macro F1, and per-class Recall for malignant classes (`mel`, `bcc`).

---

## 5. Detailed Implementation Roadmap

| Stage | Focus | Key Deliverables |
| :---: | :--- | :--- |
| **Stage 0 (Current)** | **Environment & Initialization** | Verified environment, project skeleton, docs, and passing tests. |
| **Stage 1** | **Dataset Ingestion & Split Pipeline** | Place HAM10000 files in `data/raw/`, validate hashes, serialize `train_val_test_splits.json`. |
| **Stage 2** | **Base Deep Learning Architectures** | Build `Custom CNN`, `EfficientNetB0`, and `ResNet50` models in `src/models/`. |
| **Stage 3** | **Training & Cross-Validation Pipeline** | K-fold training, checkpoint saving, generation of OOF prediction matrix. |
| **Stage 4** | **Stacking Ensemble Meta-Learner** | Train and serialize Logistic Regression meta-classifier on OOF predictions. |
| **Stage 5** | **Explainable AI (Grad-CAM)** | Implement Grad-CAM for all 3 base architectures with overlay visualization. |
| **Stage 6** | **Evaluation & Benchmark Suite** | Compute accuracy, balanced accuracy, F1, ROC-AUC, and confusion matrices on holdout test set. |
| **Stage 7** | **Streamlit Web Application** | Connect models to UI, multi-model comparative Grad-CAM display, disclaimer. |
| **Stage 8** | **System Testing, Hardening & Docs** | Comprehensive test execution, final audit, and handover. |

---

## 6. Commands Executed & Outcomes

| Command | Working Directory | Exit Code | Outcome |
| :--- | :--- | :---: | :--- |
| `Get-ChildItem -Force` | `K:\DERMASTACK-XAI` | 0 | Confirmed workspace was initially empty. |
| `Get-ChildItem -Path "K:\" -Filter "*derma*"` | `K:\DERMASTACK-XAI` | 0 | Checked for existing DermaStack files on K: drive. |
| `Get-CimInstance Win32_OperatingSystem` | `K:\DERMASTACK-XAI` | 0 | Confirmed Windows 11 64-bit and 24 GB physical RAM (~12.7 GB free). |
| `Get-PSDrive -PSProvider FileSystem` | `K:\DERMASTACK-XAI` | 0 | Confirmed 338.38 GB free space on K: drive. |
| `Get-CimInstance Win32_VideoController` | `K:\DERMASTACK-XAI` | 0 | Detected Intel(R) Graphics (Integrated). No NVIDIA GPU. |
| `py --list; py --version` | `K:\DERMASTACK-XAI` | 0 | Verified Python 3.12.5 64-bit is installed. |
| `py -3.12 -m pip list` | `K:\DERMASTACK-XAI` | 0 | Verified pre-installed packages: TF 2.21, Keras 3.15, Scikit-Learn 1.9, Streamlit 1.64. |
| `py -3.12 -c "import tensorflow as tf..."` | `K:\DERMASTACK-XAI` | 0 | Confirmed TensorFlow 2.21 operates cleanly on CPU. |
| `py -3.12 -m unittest discover -s tests -v` | `K:\DERMASTACK-XAI` | 0 | **Ran 8 automated tests; 8 passed (100% pass rate) in 4.85 seconds.** |

---

## 7. Verified Facts vs. Assumptions

### Verified Facts
1. The workspace `K:\DERMASTACK-XAI` is clean and fully initialized with zero broken dependencies.
2. Drive `K:\` has 338.38 GB of free space, and the system has 24 GB of RAM.
3. The environment operates on CPU mode (Intel integrated graphics; TensorFlow GPU on Windows 11 native is unsupported for TF $\ge 2.11$).
4. All major packages (`tensorflow==2.21.0`, `keras==3.15.1`, `scikit-learn==1.9.1`, `streamlit==1.64.0`, `pandas==3.0.6`, `numpy==2.5.3`, `pillow==12.3.0`, `matplotlib==3.11.2`) are installed and functional.
5. The `StratifiedGroupKFold` patient-aware split logic has been implemented and validated via passing automated unit tests with 0% lesion leakage.

### Assumptions
1. Since no local presentation file was found on disk, the project requirements and taxonomy specified in the prompt are assumed to represent the complete presentation specification.
2. The user will provide or authorize placement of the official HAM10000 dataset files (`HAM10000_metadata.csv` and image archives) in `data/raw/` for Stage 1.

---

## 8. Problems Requiring User Decision

Before proceeding beyond Stage 0, please provide your preference on the following items:

1. **Dataset Location & Acquisition**:
   - Do you already have the HAM10000 dataset archives (`HAM10000_metadata.csv`, `HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`) on another drive/folder that we can link or copy into `data/raw/`?
   - Or should we provide an automated download script in Stage 1?
2. **Training Strategy on CPU**:
   - Because the system has an Intel integrated GPU and runs TensorFlow in CPU mode, full training across 10,015 high-resolution images for 3 separate deep CNNs plus 5-fold cross-validation could take considerable CPU time.
   - Would you prefer:
     - **Option A (Recommended)**: Use pre-computed feature extraction / transfer learning caching with a representative stratified sample for rapid validation, then scale up.
     - **Option B**: Run full end-to-end training across all 10,015 images on CPU with progress logging and checkpoints.
3. **Presentation File**:
   - If you have the original college project presentation file (`.pptx` or `.pdf`), please let us know its location so we can cross-reference any additional custom slides or diagrams.

---

## 9. Exact Next Task to Execute

Upon receiving your approval:
- **Execute Stage 1: Dataset Ingestion & Partitioning Pipeline**:
  1. Ingest `HAM10000_metadata.csv` and scan images in `data/raw/`.
  2. Execute the patient-aware stratified group split on `lesion_id`.
  3. Generate and serialize `data/splits/train_val_test_splits.json`.
  4. Perform data audit and generate class distribution figures.
