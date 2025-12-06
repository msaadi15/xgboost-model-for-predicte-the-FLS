# FLS Prediction Using XGBoost and MSG/SEVIRI Brightness Temperatures

## Overview
This project develops two XGBoost machine learning models to detect **Fog and Low Stratus (FLS)** using thermal infrared data from the **Meteosat Second Generation (MSG) – SEVIRI** instrument combined with ground station observations.

The models predict two FLS labels:

- **`fls_station_cf`** – Clear-Fog classification  
- **`fls_station_hc`** – High-confidence fog/stratus classification  

Each model uses satellite-derived brightness temperatures, texture features, and station metadata to estimate the presence of fog or low stratus.

---

## Dataset Description

### Structure
Each row in the dataset represents a **station observation** matched with **SEVIRI brightness temperature** values at the same datetime.

Example columns:

| Column | Description |
|--------|-------------|
| `datetime` | Timestamp (UTC) of the observation |
| `station` | Meteorological station name |
| `lat`, `lon` | Station coordinates |
| `TB4`–`TB11` | Brightness temperatures from SEVIRI IR channels |
| `std_3*3_TB_X` | 3×3 neighborhood standard deviation (texture) for each BT channel |
| `fls_station_cf` | Clear-Fog classification target (0/1) |
| `fls_station_hc` | High-confidence fog/stratus label (0/1) |

### Satellite Features
The brightness temperature channels (TB4–TB11) capture thermal signatures useful for fog detection:

- **8–12 µm IR channels** → fog/stratus discrimination  
- **3.9 µm** → nighttime fog microphysics  
- **Split-window (10.8–12.0 µm)** → cloud thickness & presence of fog  
- **3×3 texture metrics** → smooth fog layers vs. broken clouds  

---

## Modeling Approach

### Two Independent Models
- **Model 1:** Predicts `fls_station_cf`
- **Model 2:** Predicts `fls_station_hc`

Both models use:
- XGBoost classifier  
- Brightness temperatures  
- Texture features  
- Station metadata  

### Workflow
1. Data preprocessing and feature engineering  
2. Train separate XGBoost classifiers  
3. Evaluate models using accuracy, recall, F1-score, and AUC  
4. Save the final models as JSON  

---

## Project Structure

data/
│ # Raw and processed datasets

models/
│ ├── xgb_fls_cf.json # Model for fls_station_cf
│ ├── xgb_fls_hc.json # Model for fls_station_hc

src/
│ ├── preprocess.py # Cleaning & feature engineering
│ ├── train_cf.py # Train model for CF
│ ├── train_hc.py # Train model for HC
│ ├── evaluate.py # Performance evaluation

notebooks/
│ ├── EDA.ipynb # Exploratory analysis
│ ├── TrainModels.ipynb # Full training workflow



