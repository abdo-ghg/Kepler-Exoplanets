# 🪐 Kepler Exoplanet Classification and Modeling Notebook

## 📘 Overview

The **Kepler Exoplanet Classification and Modeling Notebook** is part of the **ExoML** project — an AI-powered platform for exoplanet detection using machine learning.  
This notebook focuses specifically on the **Kepler dataset**, aiming to classify exoplanet candidates into two categories:

* **Confirmed / Candidate planets (1)**
* **False positives (0)**

The notebook performs **end-to-end data science tasks** — including cleaning, feature engineering, exploratory analysis, machine learning modeling, evaluation, and feature importance visualization.  
Its results are integrated with a **Flask backend** and displayed on the ExoML web dashboard, where users can visualize metrics and interact with models.

---

## 🌌 Objective

To build and evaluate multiple machine learning models that predict whether a celestial object observed by NASA’s *Kepler mission* is a **real exoplanet** or a **false detection** based on observed astrophysical parameters.

---

## 📂 Notebook Workflow

### 1. Data Loading
- Reads `Kepler.csv` from the `Data Sources/` directory.
- Inspects data shape, column names, and missing values.
- Basic statistics using `.describe()` and `.info()`.

### 2. Data Preprocessing
- Drops irrelevant or non-numeric columns (e.g., IDs, timestamps).
- Handles missing values with `fillna(mean)` or median imputation.
- Encodes categorical columns using `pd.get_dummies()`.
- Renames columns for better readability and scientific meaning.
- Creates a **Target column**:

```python
df["Target"] = np.where(df["koi_disposition"] != "FALSE POSITIVE", 1, 0)
