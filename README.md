# Student Score Prediction Pipeline
### AIAP Technical Assessment — U.A Secondary School

---

## Table of Contents
1. [Folder Structure](#1-folder-structure)
2. [Setup & Execution](#2-setup--execution)
3. [Pipeline Flow](#3-pipeline-flow)
4. [EDA Key Findings](#4-eda-key-findings)
5. [Feature Processing](#5-feature-processing)
6. [Model Selection](#6-model-selection)
7. [Model Evaluation](#7-model-evaluation)

---

## 1. Folder Structure

```
.
├── data/                    # Place score.db here — do NOT submit score.db
│   └── score.db
├── src/
│   ├── data_loader.py       # SQLite ingestion, cleaning, feature engineering
│   ├── preprocessor.py      # ColumnTransformer definition (imputation + scaling + OHE)
│   ├── models.py            # Model registry and pipeline builder
│   ├── evaluate.py          # Evaluation metrics and per-band reporting
│   └── run.py               # Main entry point — orchestrates the full pipeline
├── eda.ipynb                # Exploratory Data Analysis (Task 1)
├── config.yaml              # All configurable pipeline parameters
├── run.sh                   # Shell entry point
├── requirements.txt         # Python dependencies
└── README.md
```

Fitted model artefacts and evaluation results are written at runtime to:
```
├── models/
│   ├── gradient_boosting.pkl
│   ├── random_forest.pkl
│   └── ridge.pkl
└── results/
    └── results.csv
```

---

## 2. Setup & Execution

### Prerequisites

**Python 3.11+ (Debian/Ubuntu/macOS)** — the system Python is externally managed and blocks direct `pip install`. Use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

**Python 3.10 or earlier / Conda environments** — `pip install` works directly:

```bash
pip install -r requirements.txt
```

Place `score.db` in the `data/` folder. Do **not** submit `score.db`.

> `run.sh` automatically activates `.venv` if it exists, so after the one-time setup above you can simply run `bash run.sh` without manually activating the environment each time.

### Running the Pipeline

```bash
# Train the default model (Gradient Boosting — best CV RMSE)
bash run.sh

# Train all three models and print a comparison table
bash run.sh --model all

# Train a specific model
bash run.sh --model gradient_boosting
bash run.sh --model random_forest
bash run.sh --model ridge

# Override the database path
bash run.sh --model all --db-path data/score.db

# Use an alternate config file (e.g., for experimentation)
bash run.sh --config config_experimental.yaml
```

### Modifying Parameters

All pipeline parameters are controlled via `config.yaml`. No `.py` files need to be modified to experiment with different settings. Key configurable sections:

| Section | Parameter | Default | Purpose |
|---|---|---|---|
| `data` | `db_path` | `data/score.db` | Path to SQLite database |
| `train` | `test_size` | `0.20` | Fraction held out for final evaluation |
| `train` | `cv_folds` | `5` | Number of cross-validation folds |
| `train` | `random_state` | `42` | Reproducibility seed |
| `default_model` | — | `gradient_boosting` | Model trained when `--model` flag is absent |
| `models.gradient_boosting` | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf`, `subsample` | See below | GB hyperparameters |
| `models.random_forest` | `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf` | See below | RF hyperparameters |
| `models.ridge` | `alphas` | `[0.01..1000]` | RidgeCV candidate alphas |
| `output` | `models_dir` | `models/` | Directory for saved `.pkl` files |
| `output` | `results_dir` | `results/` | Directory for `results.csv` |

---

## 3. Pipeline Flow

```
score.db (SQLite)
      │
      ▼
data_loader.load_data()
  ├─ load_raw()          → pd.read_sql from SQLite; auto-detects table name
  ├─ clean()             → 7 cleaning steps (see §4)
  └─ engineer_features() → sleep_duration, class_size, male_ratio (see §4)
      │
      ▼
train_test_split (80/20, stratified by score band)
  └─ 5 bands: Fail(<50), C(50-59), B(60-69), A2(70-79), A1(80-100)
  └─ Ensures Fail-band students (~14% of data) appear proportionally in both sets
      │
      ▼
preprocessor.build_preprocessor()  [fitted on X_train only — no data leakage]
  ├─ Numerical (7 cols) : median imputation → StandardScaler
  ├─ Categorical (5 cols): mode imputation → OneHotEncoder(drop='first')
  └─ Binary (1 col)      : passthrough (tuition already 0/1)
      │
      ▼
models.build_pipeline(name, preprocessor, config_params)
  └─ sklearn Pipeline: preprocessor + estimator
      │
      ▼
evaluate.evaluate_model()
  ├─ Fit on X_train
  ├─ 5-fold CV RMSE on X_train (model selection criterion)
  └─ Final evaluation on X_test (RMSE, MAE, R²)
      │
      ├─ evaluate.evaluate_by_band()  → per O-level grade band breakdown
      ├─ joblib.dump() → models/<name>.pkl
      └─ results.csv  → results/<timestamp>/results.csv
```

---

## 4. EDA Key Findings

*Full analysis is in `eda.ipynb`. This section summarises the findings that directly shaped pipeline decisions.*

### Dataset Overview
- **Raw data:** 15,900 rows × 18 columns loaded from `score.db` (table: `score`)
- **After cleaning:** 14,641 rows × 13 columns (pre-engineering); 15 columns post-engineering
- **Target variable (`final_test`):** O-level mathematics score; range 32–100, mean 67.17, std 13.98, skewness +0.056 (approximately normal — no target transformation needed)

### Data Quality Issues & Resolutions

| Issue | Details | Resolution |
|---|---|---|
| Missing `final_test` | 495 rows (3.11%); missingness confirmed random (mean difference: 0.38 pts vs present) | Dropped — imputation without ground truth would contaminate labels |
| Missing `attendance_rate` | 757 rows (4.96%); confirmed MCAR (equal distribution across gender, tuition, CCA) | Median imputation in preprocessing pipeline |
| `index` column | Auto-generated row number with no predictive value | Dropped at ingestion |
| `bag_color` | The only column differing between 1,416 conflicting student records; no meaningful correlation with `final_test` in bivariate analysis | Dropped at ingestion |
| Erroneous `age` values | Values 5, 6 (missing leading `1`), −5 (sign error): 429 rows | Mapped 5→15, 6→16, −5→15; 1 row with age=−4 (irrecoverable) dropped |
| `tuition` encoding | Mixed encodings: `Yes`/`Y` and `No`/`N` | Standardised to binary int: 1 / 0 |
| `CCA` capitalisation | Mixed case: `SPORTS`, `Sports`, `sports` etc. | `.str.capitalize()` applied |

### Feature Signals (from bivariate analysis)

The five strongest signals identified in EDA, all confirmed consistent across linear and tree models:

| Feature | EDA Signal | Linear Coeff. | DT Importance | Direction |
|---|---|---|---|---|
| `class_size` (engineered) | Pearson r = −0.50 | −5.73 | 0.40 (1st) | Larger class → lower score |
| `number_of_siblings` | Strong negative box plot separation | −4.72 | 0.28 (2nd) | More siblings → lower score (less quiet study time) |
| `hours_per_week` | Negative (Simpson's paradox) | −1.18 | 0.10 (3rd) | Weaker students study *more* to compensate |
| `learning_style_Visual` | Clear box plot separation | +5.31 | 0.09 (4th) | Visual learners score higher |
| `tuition` | Consistent positive signal | +4.29 | 0.04 (6th) | Private tuition ≈ +4.3 score points |

**Notable finding — `direct_admission` sign reversal:** Raw EDA shows positive correlation with `final_test`, but after controlling for `CCA` and `class_size` in the linear model, the coefficient is −0.46. The raw correlation is a confounding artefact.

**Lasso zero-feature result:** LassoCV at optimal α=0.01 retained all 16 post-OHE features (zeroed 0/16), confirming every feature carries marginal signal. No feature elimination was applied.

### Feature Engineering

Three features were engineered from raw columns (full rationale in `eda.ipynb §6`):

| Raw Column(s) | Engineered Feature | Formula |
|---|---|---|
| `sleep_time`, `wake_time` | `sleep_duration` | `(wake_min − sleep_min) % 1440 / 60` — modulo handles overnight sleep (e.g. 23:00→06:00 = 7 h) |
| `n_male`, `n_female` | `class_size` | `n_male + n_female` |
| `n_male`, `n_female` | `male_ratio` | `n_male / class_size` |

`sleep_duration` has near-zero IQR (75%+ of students sleep exactly 8 h, std ≈ 0.60). A log transform on a near-constant distribution is unstable; `StandardScaler` handles it safely as confirmed in `prototype.ipynb §2`.

---

## 5. Feature Processing

All 13 input columns (post-cleaning, pre-engineering) are processed as follows:

| Original Column | Status | Engineered Name | Processing in Pipeline |
|---|---|---|---|
| `student_id` | Dropped before training | — | Identifier only; excluded from feature matrix |
| `index` | Dropped at ingestion | — | Auto-generated row number |
| `bag_color` | Dropped at ingestion | — | No predictive signal; sole source of conflicting duplicate records |
| `sleep_time` | Engineered → dropped | `sleep_duration` | Converted to hours via modulo arithmetic; raw column removed |
| `wake_time` | Engineered → dropped | `sleep_duration` | (see above) |
| `n_male` | Engineered → dropped | `class_size`, `male_ratio` | Used to compute class composition; raw columns removed |
| `n_female` | Engineered → dropped | `class_size`, `male_ratio` | (see above) |
| `age` | Kept → Numerical | `age` | Median imputation + StandardScaler |
| `hours_per_week` | Kept → Numerical | `hours_per_week` | Median imputation + StandardScaler |
| `attendance_rate` | Kept → Numerical | `attendance_rate` | Median imputation + StandardScaler (757 missing values, MCAR) |
| `number_of_siblings` | Kept → Numerical | `number_of_siblings` | Median imputation + StandardScaler |
| `sleep_duration` | Engineered → Numerical | `sleep_duration` | Median imputation + StandardScaler |
| `class_size` | Engineered → Numerical | `class_size` | Median imputation + StandardScaler |
| `male_ratio` | Engineered → Numerical | `male_ratio` | Median imputation + StandardScaler |
| `direct_admission` | Kept → Categorical | `direct_admission` | Mode imputation + OHE(drop='first') |
| `CCA` | Kept → Categorical | `CCA` | Mode imputation + OHE(drop='first') → 3 binary columns |
| `learning_style` | Kept → Categorical | `learning_style` | Mode imputation + OHE(drop='first') |
| `gender` | Kept → Categorical | `gender` | Mode imputation + OHE(drop='first') |
| `mode_of_transport` | Kept → Categorical | `mode_of_transport` | Mode imputation + OHE(drop='first') → 2 binary columns |
| `tuition` | Kept → Binary | `tuition` | Already 0/1 integer; passthrough (no scaling) |

**Post-OHE feature count: 16** (7 numerical + 8 OHE-expanded categorical + 1 binary)

**Scaler choice — StandardScaler over RobustScaler:** IQR outlier analysis (`eda.ipynb §10`) found 0 outliers in all numerical features except `sleep_duration`, which has 8.05% flagged purely due to its near-zero IQR (the "outliers" are legitimate students with non-standard sleep patterns). No Winsorisation or robust scaling is needed; StandardScaler is appropriate.

**OHE `drop='first'`:** Avoids the dummy variable trap for Ridge regression. Tree-based models are unaffected by the redundant reference category.

---

## 6. Model Selection

Three models were selected to represent distinct algorithmic families, each justified by the prototyping results in `prototype.ipynb`.

### Why These Three Models

The prototyping progression established a clear narrative:

| Level | Model | CV RMSE | R² | Key Finding |
|---|---|---|---|---|
| 0 | Dummy (mean) | 13.98 | 0.000 | Performance floor = std(final_test) |
| 1 | Ridge (α=10) | 9.13 | 0.588 | Linear ceiling: features explain 59% of variance linearly |
| 2 | Decision Tree (d=10) | 5.76 | 0.841 | Non-linearity: R² jumps 0.59→0.84; single tree still overfits |
| 3 | Random Forest (tuned) | 5.36 | 0.854 | Bagging closes overfit gap from −7.40 to −2.20 |
| 3 | Gradient Boosting (tuned) | **5.35** | **0.854** | Boosting achieves best CV RMSE with smallest overfit gap |

The 3.76-point RMSE gap between Ridge (9.13) and the ensembles (5.35) quantifies the non-linearity premium in this dataset — the dominant features (`class_size`, `number_of_siblings`, `hours_per_week`) interact non-linearly with `final_test`, as confirmed by the Decision Tree's 40% Gini importance concentration in `class_size` alone.

### Model 1 — Ridge Regression (`ridge`)
**Purpose:** Interpretable linear baseline.

Ridge with L2 regularisation was chosen over plain OLS because it includes built-in cross-validated alpha selection (`RidgeCV`), providing a rigorous baseline. The improvement over OLS was negligible (−0.0003 RMSE), confirming no multicollinearity issue. Lasso at α=0.01 zeroed 0 of 16 features, confirming Ridge's full feature set is appropriate.

Ridge provides the only model with directly interpretable coefficients — `class_size` (−5.73), `learning_style_Visual` (+5.31), `number_of_siblings` (−4.72), `tuition` (+4.29) — directly answering which student factors matter most, serving the school's goal of identifying at-risk students.

**Tuned hyperparameter:** `alpha = 10.0` (selected by inner 5-fold CV from candidates [0.01, 0.1, 1, 10, 100, 1000])

### Model 2 — Random Forest Regressor (`random_forest`)
**Purpose:** Bagging ensemble; strong non-linear predictor.

Random Forest addresses the single-tree overfit problem by averaging 326 trees trained on bootstrap samples with random feature subsets (`max_features=0.5`). The focused `min_samples_leaf` sweep (values 1–10) confirmed msl=1 is the globally optimal setting — CV RMSE increases monotonically at every step from 1 to 10. This is possible because the averaging of 326 trees provides sufficient variance reduction to handle fully-grown individual leaves.

**Tuned hyperparameters** (via `RandomizedSearchCV`, 30 iterations):

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 326 | More trees → lower variance; diminishing returns beyond ~300 |
| `max_depth` | 15 | Deeper than optimal single-tree depth (10) because ensemble averaging compensates for individual tree variance |
| `max_features` | 0.5 | 8 of 16 features per split; better than sqrt=4 for small feature sets (maintains sufficient tree diversity) |
| `min_samples_leaf` | 1 | Confirmed globally optimal by focused sweep (msl=1→10 monotonically degrades CV RMSE) |

### Model 3 — Gradient Boosting Regressor (`gradient_boosting`) ← Primary Model
**Purpose:** Boosting ensemble; best generalisation performance.

Gradient Boosting builds trees sequentially, each correcting the residual errors of the previous, reducing bias through iterative refinement. Unlike Random Forest, it directly targets the loss function on each round. After three rounds of tuning (50 iterations each with `RandomizedSearchCV`), the final configuration achieves the best CV RMSE of all models (5.3456) and the smallest overfit gap (~0.75 vs RF's 2.20), indicating more stable generalisation to new student cohorts.

The low learning rate (`lr=0.01`) with more rounds (`n_estimators=548`) was preferred over the faster v1 configuration (`lr=0.03`, n=401) because it finds a more regularised solution. Deeper trees (`max_depth=8`) combined with higher leaf constraints (`min_samples_leaf=6`) balance expressiveness and regularisation.

**Tuned hyperparameters** (final configuration — v2, preferred over v3 for lower overfit gap):

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 548 | Sufficient rounds for convergence at lr=0.01 |
| `learning_rate` | 0.01 | Slow updates → more regularised solution; compensated by higher n_estimators |
| `max_depth` | 8 | Captures non-linear threshold effects in `class_size` and `number_of_siblings` |
| `min_samples_leaf` | 6 | Leaf regularisation balances expressiveness of depth=8 |
| `subsample` | 0.8 | Stochastic GB: 80% of training samples per round reduces tree correlation |

---

## 7. Model Evaluation

### Metrics

All three metrics are reported on the held-out test set (2,929 rows, 20% of data):

| Metric | Formula | Why Used |
|---|---|---|
| **RMSE** (primary) | √(Σ(yᵢ − ŷᵢ)² / n) | Penalises large errors more than MAE. A large mis-prediction on a borderline Fail-band student (<50) has real consequences — the student would not receive intervention. Primary criterion for model selection. |
| **MAE** | Σ\|yᵢ − ŷᵢ\| / n | Mean absolute error in the same unit as `final_test` (0–100 scale). Interpretable to non-technical stakeholders: "on average, predictions are off by X score points." |
| **R²** | 1 − SS_res / SS_tot | Proportion of variance in `final_test` explained by the model. Communicates overall model quality; R²=0 means the model is no better than predicting the mean. |
| **CV RMSE** | 5-fold CV on training set | Used for model selection and hyperparameter tuning. Prevents test-set leakage — the test set is touched only once for final reporting. |

No target transformation (log, sqrt) was applied. `final_test` has skewness = +0.056 (near-normal), so the raw scale RMSE and MAE are interpretable directly as score-point errors.

### Results

| Model | CV RMSE | CV std | Test RMSE | Test MAE | Test R² | Overfit Gap |
|---|---|---|---|---|---|---|
| Dummy (mean baseline) | 13.98 | 0.09 | 13.99 | 11.66 | 0.000 | −0.01 |
| Ridge (α=10) | 9.1312 | 0.1105 | 8.9819 | 7.1297 | 0.5880 | +0.1353 |
| **Gradient Boosting (tuned)** | **5.3465** | **0.0941** | **5.2794** | **3.6936** | **0.8577** | **−0.9219** |
| Random Forest (tuned) | 5.3597 | 0.1039 | 5.3475 | 3.6990 | 0.8540 | −2.2041 |

### Interpretation

All three models beat the Dummy baseline by a wide margin, confirming that the 13 features carry genuine predictive signal. The ensemble models improve over Ridge by **3.70 RMSE points** (GB) — this gap quantifies the non-linearity that linear models structurally cannot capture. GB reduces the Dummy baseline RMSE by **62.3%** (13.99 → 5.28); Ridge by 35.8%.

The Gradient Boosting model is recommended for deployment as the primary model because:
1. It has the best CV RMSE (5.3465) and Test RMSE (5.2794) of all three models
2. Its overfit gap (−0.9219) is **2.4× smaller** than Random Forest (−2.2041), meaning predictions will degrade less when applied to new student cohorts
3. Its CV std (0.0941) is also the smallest, indicating the most consistent performance across folds
4. Its sequential boosting mechanism directly minimises prediction error on the hardest-to-predict students — exactly the Fail-band students the school needs to identify

Ridge is retained as an interpretable secondary model: its coefficients provide actionable insights (e.g., the +4.29 coefficient on `tuition` directly quantifies the intervention value of private tuition).

Note: RF's large Train−Test gap (−2.20) reflects that min_samples_leaf=1 allows individual trees to approach zero training error, which is an expected property of deep random forests. The CV−Test gap of +0.01 confirms this does not affect generalisation.

### Per-Band Evaluation

The pipeline reports per-band RMSE and MAE breakdown across all five O-level grade bands at runtime. This directly addresses the business objective: a model acceptable on overall RMSE but with high Fail-band (<50) RMSE would fail at identifying weaker students. Run `bash run.sh --model all` to see the full breakdown in the console output.
