# AI-Based Anomaly Detection and Intelligent Monitoring of Chiller Operating Data

## Academic Metadata
- **Student Register Number:** 24BCE0551
- **Track:** Multivariate and Time-Series Detection
- **Assigned Method:** PCA-Based Multivariate Detector (Reconstruction Error / SPE)
- **Primary Research Question:** *Can anomalies be detected from correlated temperature and pressure variables?*
- **Assigned Preprocessing:** Imputation, StandardScaler, Correlation Analysis
- **Evaluation:** Reconstruction Error, Synthetic Anomaly Testing (Spike, Drift, Correlation Breaking)

---

## Project Structure
```text
Chiller-Anomaly-Detection/
│
├── data/
│   ├── raw/
│   │   └── official_dataset/      # Preserved unmodified raw dataset
│   ├── processed/                 # Chronologically split, scaled features
│   └── synthetic/                 # Clean test copy with injected anomalies
│
├── notebooks/
│   └── Chiller_PCA_Anomaly_Detection.ipynb
│
├── src/
│   ├── data_inspection.py         # Phase 1 dataset inspection & profiling
│   ├── preprocessing.py           # Phase 3 feature selection, split & scaling
│   ├── pca_detector.py            # Phase 4 PCA model training & variance
│   ├── anomaly_detection.py       # Phase 5 SPE calculation & thresholding
│   └── evaluation.py              # Phase 6 & 7 synthetic evaluation & metrics
│
├── results/
│   ├── figures/                   # Visualizations & diagnostic plots
│   ├── metrics/                   # Tabulated performance metrics
│   └── final_analysis/            # Quantitative summaries
│
├── documentation/
│   ├── ai_assistance_log.md       # Academic integrity AI assistance audit log
│   └── dataset_profile_working.md # Working dataset facts & engineering notes
│
├── presentation/                  # Presentation slides & viva materials
│
├── requirements.txt               # Python package dependencies
├── README.md                      # Project overview and instructions
└── .gitignore
```

---

## Setup & Environment
1. Ensure Python 3.10+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run dataset inspection:
   ```bash
   python src/data_inspection.py
   ```

