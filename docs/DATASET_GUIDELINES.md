# HAM10000 Dataset Guidelines & Leakage Prevention Protocol

## 1. Official Dataset Overview & Source

The **HAM10000** ("Human Against Machine with 10000 training images") dataset is a landmark benchmark collection of dermatoscopic images of pigmented skin lesions collected from the Medical University of Vienna (Austria) and the Skin Cancer Practice in Queensland (Australia).

### Official Publication Citation
> Tschandl, P., Rosendahl, C. & Kittler, H.  
> **The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.**  
> *Scientific Data* 5, 180161 (2018).  
> DOI: [10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

### Official Download Repositories
1. **Harvard Dataverse**:
   - URL: `https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T`
   - Archive Files:
     - `HAM10000_images_part_1.zip` (~1.9 GB)
     - `HAM10000_images_part_2.zip` (~1.4 GB)
     - `HAM10000_metadata.csv` (approx 580 KB)
2. **ISIC Archive / Kaggle**:
   - URL: `https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000`
   - Contains the exact identical images and metadata file.

> [!IMPORTANT]
> In accordance with project instructions, large dataset archives must NOT be silently downloaded without user confirmation. When ready to acquire the dataset, download the files into `data/raw/` or extract them as instructed below.

---

## 2. Directory Layout for Ingestion

The raw dataset should be placed under `data/raw/` in the project root:

```text
k:\DERMASTACK-XAI\data\raw\
  ├── HAM10000_metadata.csv
  ├── HAM10000_images_part_1\
  │     ├── ISIC_0024306.jpg
  │     └── ...
  └── HAM10000_images_part_2\
        ├── ISIC_0029306.jpg
        └── ...
```
Alternatively, all images may be consolidated into a single subfolder `data/raw/images/`. The dataset loader in `src/data/dataset.py` will search both subdirectories automatically.

---

## 3. Metadata Schema Specification

The master metadata file `HAM10000_metadata.csv` consists of 10,015 rows and the following 7 columns:

| Column | Data Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `lesion_id` | `str` | Biological lesion identifier. Multiple images may share the same `lesion_id`. | `HAM_0000118` |
| `image_id` | `str` | Unique dermatoscopic image identifier. | `ISIC_0027419` |
| `dx` | `str` | Ground truth diagnosis (one of 7 categories). | `bkl` |
| `dx_type` | `str` | Confirmation ground truth type: `histo` (histopathology), `follow_up` (digital dermatoscopy follow-up), `consensus` (expert consensus), `confocal` (in-vivo confocal microscopy). | `histo` |
| `age` | `float` | Patient age in years (contains missing values; median imputation recommended). | `80.0` |
| `sex` | `str` | Patient sex (`male`, `female`, `unknown`). | `male` |
| `localization` | `str` | Anatomical body location (e.g. `back`, `lower extremity`, `trunk`, `face`). | `scalp` |

---

## 4. Critical Data Leakage Risks & Prevention Protocol

### 4.1 The Lesion-Level Duplicate Problem
In the HAM10000 dataset:
- Total Images: **10,015**
- Unique Lesions (`lesion_id`): **7,470**
- Duplicate Ratio: Approximately **25.4%** of images are alternate views (different lighting, magnification, or timepoints) of an already present lesion.

> [!WARNING]
> **Catastrophic Leakage Failure Mode**:  
> If an unstratified or standard random `train_test_split` is applied across image rows, identical lesions will be assigned to both the training set and the test set. Because the network easily memorizes the unique visual patterns of the lesion, this causes **severe artificial metric inflation** (e.g., reporting 90%+ accuracy when real clinical generalization is < 75%).

### 4.2 Mandatory Leakage Prevention Protocol
To guarantee complete clinical and statistical validity:
1. **Grouping**: The splitting algorithm MUST group by `lesion_id`.
2. **Stratified Group Split**:
   - `StratifiedGroupKFold` or custom patient-aware stratified splitting must be used so that:
     $$\text{Lesions}(\text{Train}) \cap \text{Lesions}(\text{Val}) = \emptyset$$
     $$\text{Lesions}(\text{Train}) \cap \text{Lesions}(\text{Test}) = \emptyset$$
     $$\text{Lesions}(\text{Val}) \cap \text{Lesions}(\text{Test}) = \emptyset$$
3. **Automated Leakage Audit**:
   - An automated test (`tests/test_data_leakage.py`) must verify that no lesion identifier appears in more than one partition before any model training begins.
4. **Out-of-Fold (OOF) Training for Stacking Meta-Learner**:
   - When training the stacking Logistic Regression meta-model, base model prediction inputs MUST be out-of-fold predictions. Training the meta-learner on in-sample base model predictions causes overfitting and destroys ensemble generalization.

---

## 5. Class Imbalance Characteristics

The 10,015 images are distributed across the 7 classes as follows:

| Class Code | Category | Image Count | Approximate Frequency |
| :--- | :--- | :--- | :--- |
| `nv` | Melanocytic nevi | 6,705 | **66.95%** |
| `mel` | Melanoma | 1,113 | **11.11%** |
| `bkl` | Benign keratosis-like | 1,099 | **10.97%** |
| `bcc` | Basal cell carcinoma | 514 | **5.13%** |
| `akiec` | Actinic keratoses | 327 | **3.27%** |
| `vasc` | Vascular lesions | 142 | **1.42%** |
| `df` | Dermatofibroma | 115 | **1.15%** |

### Imbalance Mitigation Strategy
- **Balanced Class Weights**: Calculate inverse frequency class weights during training:
  $$w_c = \frac{N}{K \cdot N_c}$$
- **Evaluation Metric Focus**: Primary evaluation must emphasize **Balanced Accuracy**, **Macro F1-Score**, and **Per-Class Recall** (especially for dangerous malignancies like `mel` and `bcc`), rather than raw accuracy alone.
