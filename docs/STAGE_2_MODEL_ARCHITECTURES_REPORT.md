# DermaStack-XAI — Stage 2 Execution Report
**Base Deep Learning Architectures and Forward-Pass Validation**

**Date & Time**: 2026-10-18  
**Workspace**: `K:\DERMASTACK-XAI`  
**Status**: Stage 2 Complete (All 3 Base Architectures Implemented, Verified & Tested)

---

## 1. Inspection & Environment Audit Prior to Implementation

### Verified Environment Details:
- **Operating System**: Microsoft Windows 11 Home Single Language (64-bit, Version `10.0.26200`).
- **Python Version**: **3.12.5 (64-bit)**.
- **TensorFlow Version**: **2.21.0** (running in native Windows CPU mode: `/physical_device:CPU:0`).
- **Keras Version**: **3.15.1** (Keras 3 with TensorFlow backend).
- **Physical Memory (RAM)**: **24 GB** total (~12.7 GB free).
- **Disk Storage**: Drive `K:\` has **333.19 GB free**.
- **Stage 1 Status**: Fully preserved (`data/raw/` with 10,015 images, `data/processed/` with validated metadata and audit JSON, `data/splits/train_val_test_splits.json`).

---

## 2. Files Inspected, Created, or Modified

| File Path | Nature | Description |
| :--- | :---: | :--- |
| [`src/models/__init__.py`](file:///k:/DERMASTACK-XAI/src/models/__init__.py) | **Created** | Public package interface exposing model builders and utilities. |
| [`src/models/common.py`](file:///k:/DERMASTACK-XAI/src/models/common.py) | **Created** | Shared utilities: `get_model_summary_dict`, `preprocess_for_model`, and `validate_output_tensor`. |
| [`src/models/custom_cnn.py`](file:///k:/DERMASTACK-XAI/src/models/custom_cnn.py) | **Created** | 4-block lightweight CNN (< 500K parameters) with `conv4_block` Grad-CAM layer. |
| [`src/models/efficientnet.py`](file:///k:/DERMASTACK-XAI/src/models/efficientnet.py) | **Created** | EfficientNetB0 transfer learning model with custom 7-class head and backbone freeze toggle. |
| [`src/models/resnet.py`](file:///k:/DERMASTACK-XAI/src/models/resnet.py) | **Created** | ResNet50 transfer learning model with custom 7-class head and backbone freeze toggle. |
| [`tests/test_models.py`](file:///k:/DERMASTACK-XAI/tests/test_models.py) | **Created** | 11 unit & integration tests covering construction, shapes, bounds, trainability, Grad-CAM, serialization. |
| [`docs/STAGE_2_MODEL_ARCHITECTURES_REPORT.md`](file:///k:/DERMASTACK-XAI/docs/STAGE_2_MODEL_ARCHITECTURES_REPORT.md) | **Created** | This comprehensive execution report. |
| [`docs/IMPLEMENTATION_CHECKLIST.md`](file:///k:/DERMASTACK-XAI/docs/IMPLEMENTATION_CHECKLIST.md) | **Updated** | Stage 2 status marked as completed. |
| [`README.md`](file:///k:/DERMASTACK-XAI/README.md) | **Updated** | Updated test badge from 22 Passing to 33 Passing. |

---

## 3. Architecture Summaries & Parameter Accounting

### Model 1: Custom Lightweight CNN (`src/models/custom_cnn.py`)
- **Philosophy**: Parameter-efficient, fast-training convolutional network designed to serve as a low-complexity baseline and ensemble contributor.
- **Architecture**:
  - `InputLayer`: `(224, 224, 3)`
  - `Block 1`: `Conv2D(32, 3x3, same)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2D(2x2)` $\rightarrow$ `(112, 112, 32)`
  - `Block 2`: `Conv2D(64, 3x3, same)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2D(2x2)` $\rightarrow$ `(56, 56, 64)`
  - `Block 3`: `Conv2D(128, 3x3, same)` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2D(2x2)` $\rightarrow$ `(28, 28, 128)`
  - `Block 4`: `Conv2D(256, 3x3, same, name="conv4_block")` $\rightarrow$ `BatchNorm` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2D(2x2)` $\rightarrow$ `(14, 14, 256)`
  - `Head`: `GlobalAveragePooling2D()` (256) $\rightarrow$ `Dense(128, relu)` $\rightarrow$ `BatchNorm` $\rightarrow$ `Dropout(0.4)` $\rightarrow$ `Dense(7, softmax)`
- **Grad-CAM Target Layer**: `"conv4_block"` (spatial output: `(None, 28, 28, 256)`).
- **Exact Parameter Counts**:
  - Total Parameters: **424,647** (< 2,000,000 budget, **PASSED**)
  - Trainable Parameters: **423,431**
  - Non-Trainable Parameters: **1,216** (BatchNorm statistics)

---

### Model 2: EfficientNetB0 (`src/models/efficientnet.py`)
- **Philosophy**: Compound-scaled neural architecture with inverted residual blocks (MBConv), achieving high accuracy with minimal floating-point operations.
- **Architecture**:
  - `Backbone`: `keras.applications.EfficientNetB0(include_top=False, input_shape=(224, 224, 3))`
  - `Head`: `GlobalAveragePooling2D(name="head_gap")` $\rightarrow$ `BatchNorm(name="head_bn")` $\rightarrow$ `Dropout(0.4, name="head_dropout")` $\rightarrow$ `Dense(7, softmax, name="predictions")`
- **Grad-CAM Target Layer**: `"top_conv"` (spatial output: `(None, 7, 7, 1280)`).
- **Exact Parameter Counts**:
  - Total Parameters: **4,063,658**
  - **With Backbone Frozen (`freeze_backbone=True`)**:
    - Trainable Parameters: **11,527** (custom classification head)
    - Non-Trainable Parameters: **4,052,131** (frozen backbone)
  - **With Backbone Unfrozen (`freeze_backbone=False`)**:
    - Trainable Parameters: **4,019,075**
    - Non-Trainable Parameters: **44,583**

---

### Model 3: ResNet50 (`src/models/resnet.py`)
- **Philosophy**: Deep residual architecture utilizing identity skip connections to enable effective feature extraction across 50 layers without vanishing gradients.
- **Architecture**:
  - `Backbone`: `keras.applications.ResNet50(include_top=False, input_shape=(224, 224, 3))`
  - `Head`: `GlobalAveragePooling2D(name="head_gap")` $\rightarrow$ `Dense(256, relu, name="head_dense")` $\rightarrow$ `BatchNorm(name="head_bn")` $\rightarrow$ `Dropout(0.5, name="head_dropout")` $\rightarrow$ `Dense(7, softmax, name="predictions")`
- **Grad-CAM Target Layer**: `"conv5_block3_out"` (spatial output: `(None, 7, 7, 2048)`).
- **Exact Parameter Counts**:
  - Total Parameters: **24,115,079**
  - **With Backbone Frozen (`freeze_backbone=True`)**:
    - Trainable Parameters: **526,855** (custom classification head)
    - Non-Trainable Parameters: **23,588,224** (frozen backbone)
  - **With Backbone Unfrozen (`freeze_backbone=False`)**:
    - Trainable Parameters: **24,061,447**
    - Non-Trainable Parameters: **53,632**

---

## 4. Input Shapes, Preprocessing & Output Contract

| Specification | Custom CNN | EfficientNetB0 | ResNet50 |
| :--- | :--- | :--- | :--- |
| **Input Shape** | `(batch_size, 224, 224, 3)` | `(batch_size, 224, 224, 3)` | `(batch_size, 224, 224, 3)` |
| **Input Data Type** | `float32` | `float32` | `float32` |
| **Preprocessing Function** | `images / 255.0` (scales raw $[0, 255] \rightarrow [0.0, 1.0]$) | `keras.applications.efficientnet.preprocess_input` (expects $[0, 255]$ RGB floats; internal layers handle normalization) | `keras.applications.resnet50.preprocess_input` (converts RGB $\rightarrow$ BGR, subtracts ImageNet channel means: $[-123.68, +151.06]$) |
| **Output Shape** | `(batch_size, 7)` | `(batch_size, 7)` | `(batch_size, 7)` |
| **Output Activation** | Softmax | Softmax | Softmax |
| **Output Bounds** | $[0.0, 1.0]$, sums to $1.0 \pm 10^{-4}$ | $[0.0, 1.0]$, sums to $1.0 \pm 10^{-4}$ | $[0.0, 1.0]$, sums to $1.0 \pm 10^{-4}$ |
| **Class Index Order** | `[akiec, bcc, bkl, df, mel, nv, vasc]` | `[akiec, bcc, bkl, df, mel, nv, vasc]` | `[akiec, bcc, bkl, df, mel, nv, vasc]` |

---

## 5. Pretrained ImageNet Weights Status

- **Status**: **Genuinely verified and tested**.
  - Verified presence of cached weights in `C:\Users\HP\.keras\models\`:
    - `efficientnetb0_notop.h5` (`16,705,208` bytes)
    - `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` (`94,765,736` bytes)
  - An explicit integration test (`test_pretrained_imagenet_weights_loading_integration`) validated that genuine ImageNet weights load locally with zero network access required.
  - Unit tests by default run with `weights=None` for rapid, deterministic execution.

---

## 6. Serialization & Grad-CAM Accessibility Results

- **Serialization Testing (`.keras` format)**:
  - All three models were serialized to temporary `.keras` archives and reloaded using `keras.models.load_model()`.
  - Predictions on dummy batches before and after reload were compared using `np.testing.assert_allclose(atol=1e-5)`.
  - **Maximum Absolute Discrepancy**: **0.0** (Perfect numerical fidelity; no anonymous lambda layers or custom layer issues).
- **Grad-CAM Target Feature Maps**:
  - `custom_cnn`: `"conv4_block"` produces a 4D tensor `(None, 28, 28, 256)`.
  - `efficientnet_b0`: `"top_conv"` produces a 4D tensor `(None, 7, 7, 1280)`.
  - `resnet50`: `"conv5_block3_out"` produces a 4D tensor `(None, 7, 7, 2048)`.
  - All target layers are directly retrievable via `model.get_layer(name)`.

---

## 7. Test Results Summary

### Model Architecture Test Command:
```powershell
py -3.12 -m unittest tests.test_models -v
```
**Results**: **11/11 PASSED** (18.10s)

### Full Test Suite Discovery Command:
```powershell
py -3.12 -m unittest discover -s tests -p "test_*.py" -v
```
**Results**:
- **Total Tests**: **33**
- **Passed**: **33 (100% pass rate)**
- **Failures / Errors**: **0**
- **Runtime**: **21.729 seconds**

#### Test Breakdown:
- `tests/test_models.py`: 11 tests (All 3 models built, forward passes on CPU, shapes, parameter bounds, trainability controls, Grad-CAM feature maps, save/reload fidelity, ImageNet loading).
- `tests/test_data_leakage.py`: 9 tests (Patient-level & lesion-level grouping, zero overlap, split reproducibility, rare classes).
- `tests/test_dataset_validation.py`: 8 tests (Corrupted files, duplicate IDs, missing files, invalid labels, artifact saving).
- `tests/test_environment.py`: 5 tests (Python 3.12, TensorFlow 2.21, Keras 3, seeds, paths).

---

## 8. Unresolved Issues & Resource Risks

- **Resource Notice**: As established, the host operates in CPU mode (Intel Integrated Graphics). The models have been configured with frozen backbones by default for initial transfer learning, which reduces trainable parameters from 4.06M to **11.5K** for EfficientNetB0 and from 24.1M to **526.8K** for ResNet50. This design ensures that CPU training during Stage 3 will remain memory-safe and computationally feasible.
- **Zero Unresolved Errors**: All models, shared utilities, tests, and configurations are verified and operational.
