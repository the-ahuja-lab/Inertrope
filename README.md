#  Inertrope

**Inertrope** is a machine learning–driven multiclass framework designed to classify patient data from **Isothermal Titration Calorimetry (ITC)** and **Spectroscopic measurements** into **Healthy**, **Benign**, and **Cancer** classes.  
It integrates systematic normalization, feature engineering, and advanced classification modeling to provide a **robust, interpretable**, and **scalable** platform for clinical diagnostics.

---

<br>
<div align="center">
  <img src="Images/inertrope.png" alt="Inertrope" width="750">
</div>
<br>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg">
  <img src="https://img.shields.io/badge/docs-passing-green">
  <img src="https://img.shields.io/badge/python-3.9+-blue">
  <a href="https://github.com/the-ahuja-lab/inertrope">
    <img src="https://img.shields.io/badge/Code-Source-black">
  </a>
</p>

---
---

## ⚙️ Installation

### 🧱 Using Conda

```bash
conda create -n inertrope -c conda-forge python=3.9
conda activate inertrope
```

### 📦 Required Dependencies

- `pandas >= 1.24.4`
- `numpy >= 1.24.3`
- `scikit-learn == 1.3.0`
- `tsfresh == 0.20.3`
- `tqdm`
- `joblib`




### 🧪 From Source

```bash
git clone https://github.com/the-ahuja-lab/inertrope.git
cd inertrope
pip install -e .
```
### 🧪 Using pip

```python
!pip install git+https://github.com/the-ahuja-lab/Inertrope.git
```
---

## 🔑 License Key

- **Academic use** – Free for research and educational institutions  
- **Commercial use** – Requires a license key (contact us for details)

---

## 🚀 Workflow Overview

The Inertrope pipeline consists of **three main steps**:

### 1️⃣ ITC Data Normalization
- Baseline correction for ITC and Absorbance data 


### 2️⃣ Feature Extraction from ITC Differential Power (DP μ/sec) Calorimetric Fingerprints  
- Time-series features via `tsfresh`
- Log Normalization    

### 3️⃣ Inference of patient clinical status using Multiclass Model trained on 1.) ITC data  2.) ITC and UV-Vis Absorbance data of Healthy, Benign, and Cancer patients' plasma with inertrope.
- Classes: *Healthy*, *Benign*, *Cancer*   

---

## 🧬 Quick Start
```python

from Inertrope import inertrope

## Configure Muticlass Model paths  

inertrope.configure_models(
    itc_model_path="/path/final_extratrees_combined.joblib",
    combined_model_path="/path/final_extratrees_combined.joblib"
)
```
### 🧪 Predict from ITC Data

```python

results_itc = inertrope.pred_itc("ITC_normalized_data.csv", out_csv="itc_predictions.csv")
```

### 🔬 Predict from Combined ITC + UV-Vis Spectroscopy Data (200–900 nm)

```python
results_combined = inertrope.pred_combined("ITC_normalized_combined_data.csv", out_csv="combined_predictions.csv")
```

---

## 📊 Output Format

Predictions are returned as a pandas DataFrame:

| Sample_Id | Prediction | Prob_Healthy | Prob_Benign | Prob_Cancer |
|------------|-------------|---------------|---------------|--------------|
| P001 | Healthy | 0.91 | 0.06 | 0.03 |
| P002 | Cancer  | 0.02 | 0.10 | 0.88 |

An output CSV file (`*_predictions.csv`) will also be generated if `out_csv` is specified.

---

