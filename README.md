# Inertrope

Inertrope is a machine learning–driven multiclass framework developed to categorize patient data from Isothermal Titration Calorimetry (ITC) and Spectroscopic measurements into Healthy, Benign, and Cancer classes. The pipeline integrates systematic normalization, feature engineering, and advanced classification modeling, providing a robust and interpretable platform for clinical diagnostics.

---


<br>
<div align="center">
<img src="Images/inertrope.png" alt="Inertrope" ></div>
<br>

<div align="left">

<div align="left">

<p>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg">
  <img src="https://img.shields.io/badge/docs-passing-green">
  <img src="https://img.shields.io/badge/python-3.9-blue">
  </a>
  <a href="https://github.com/YOUR_USERNAME/YOUR_REPO">
    <img src="https://img.shields.io/badge/Code-Source-black">
  </a>
</p>

</div>


## Introduction

Isothermal Titration Calorimetry (ITC) generates thermodynamic binding data that often require careful normalization before they can be reliably used for downstream classification. Inertrope addresses this by:

1. **Normalizing ITC data** to correct baseline drift and batch effects.
2. **Extracting features** relevant to differential thermodynamic profiles.
3. **Training a multiclass ML model** capable of discriminating between *Healthy*, *Benign*, and *Cancer* cohorts.

This stepwise approach ensures reproducibility, clinical interpretability, and scalability.

---
```bash
conda create -c conda-forge -n inertrope
conda activate inertrope

```

### Major dependencies

- **pandas** (v1.24.4)
- **tsfresh** (v0.20.3) (time-series feature extraction for ITC thermograms)
- **numpy** (v1.24.3)
- **scikit-learn (v1.3.0)**
- **matplotlib**
- **tqdm**
- **joblib**

Install directly:

```bash
pip install pandas numpy scikit-learn==1.3.0 matplotlib tqdm joblib  tsfresh 

```

## Installation

For development mode (from source):

```bash
git clone https://github.com/the-ahuja-lab/inertrope.git
cd inertrope
pip install -e .

```

---

## License Key

- **Academic use**: Free for research and educational institutions.
- **Commercial use**: Requires a license key. Contact us for details.

---

## Workflow Overview

The Inertrope pipeline consists of **three major steps**:

1. **ITC Data Normalization**
    - Baseline correction
    - Heat per injection normalization
    - Replicate merging
2. **Feature Extraction**
    - Thermodynamic descriptors
    - Statistical descriptors (mean, variance, slope)
    - tsfresh time-series features
3. **Multiclass Model Training**
    - Targets: **Healthy**, **Benign**, **Cancer**
    - Models: RandomForest, XGBoost, Logistic Regression
    - Cross-validation with metrics (Accuracy, F1, AUC)
    - Export trained model as `.pkl`

---

## Quick Start for Inference of patient clinical status into 3 classes (Healthy, Benign and Cancer)  

```bash

from Inertrope import inertrope

```

## Using Isothermal Calorimetry Data for plasma samples

```bash
results_ITC = inertrope.pred_itc("example_itc_data.csv", out_csv="itc_predictions.csv")
```

## Using Isothermal Calorimetry Data and UV-Visible Wavelength (200-900nm) for plasma samples

```bash
results_combined = inertrope.pred_combined("example_combined_data.csv", out_csv="combined_predictions.csv")
```


```

## Outputs
pandas Dataframe of containing
|Sample_Id | Prediction | Prob_Healthy | Prob_Benign | Prob_Cancer


