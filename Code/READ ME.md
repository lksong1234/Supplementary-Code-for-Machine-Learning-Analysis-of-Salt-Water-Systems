```markdown
# Supplementary Code for Machine Learning Analysis of Salt–Water Systems

## 1. Overview

This repository contains the Python scripts used for the machine learning
analysis presented in the manuscript and its Supplementary Information.
Two independent tasks are included:

1. **Regression task** — multi-algorithm comparison for predicting the
   concentration of component 2 (C₂) in salt–water systems, including
   5-fold cross-validation, interpolation testing, and high-temperature
   extrapolation testing.
2. **Classification task** — 5-fold stratified cross-validation for
   predicting the equilibrium solid phase, using a data-leakage-free
   pipeline.

All figures and result tables are generated automatically and saved to
the corresponding folders under `results/`.

## 2. System Requirements

- Python 3.8 or higher
- Required packages:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - openpyxl

Install the required packages with:

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

## 3. Repository Structure

```
project_root/
├── data/
│   └── data.xlsx                              # Input dataset (required)
├── results/
│   ├── classification results/                # Output of the classification task
│   └── multi_algorithm_regression_results/    # Output of the regression task
├── src/
│   ├── classification task/
│   │   └── main.py                            # Classification script
│   └── regression task/
│       └── main.py                            # Regression script
└── README.md
```

- `data/data.xlsx` — input dataset for all systems.
- `results/classification results/` — automatically generated outputs of the
  classification task.
- `results/multi_algorithm_regression_results/` — automatically generated
  outputs of the regression task.
- `src/classification task/main.py` — classification script.
- `src/regression task/main.py` — regression script.

Each script resolves paths relative to its own location, so it can be
executed from any working directory.

## 4. Input Data Format

The input file `data/data.xlsx` must be an Excel workbook with one worksheet
per chemical system (e.g., `NaCl-KCl-H₂O`, `Na₂SO₄-MgSO₄-H₂O`).

Each worksheet must contain the following columns (column names are matched
by keywords, so both Chinese and English names are supported):

| Column | Description | Example header |
|--------|-------------|----------------|
| Temperature | Temperature (K or °C) | `温度` or `Temperature` |
| Component 1 | Concentration of component 1 | `组分1` or `Component1` |
| Component 2 | Concentration of component 2 | `组分2` or `Component2` |
| Solid phase | Equilibrium solid phase label | `固相` or `Equilibrium` |

Missing values are removed automatically before analysis.

## 5. How to Run

Run each task from the project root:

**Regression task:**

```bash
python "src/regression task/main.py"
```

**Classification task:**

```bash
python "src/classification task/main.py"
```

The scripts will create the output folders automatically if they do not exist.

## 6. Output Description

### 6.1 Regression Task

All results are saved to `results/multi_algorithm_regression_results/`.

| File | Description |
|------|-------------|
| `data/cv5_metrics_by_fold_all_models.xlsx` | Per-fold cross-validation metrics for all algorithms |
| `data/cv5_metrics_summary_all_models.xlsx` | Mean and standard deviation of cross-validation metrics |
| `data/interpolation_metrics_all_models.xlsx` | Interpolation test metrics for all algorithms |
| `data/extrapolation_metrics_all_models.xlsx` | High-temperature extrapolation metrics for all algorithms |
| `data/interpolation_prediction_data_all_models.xlsx` | Predicted vs. experimental values (interpolation) |
| `data/extrapolation_prediction_data_all_models.xlsx` | Predicted vs. experimental values (extrapolation) |
| `data/best_model_summary_by_cv.xlsx` | Best-performing algorithm for each system |
| `figures/<system_name>/*.png` | SCI-style experimental vs. predicted scatter plots |
| `figures/algorithm_comparison/*.png` | Performance comparison bar charts |

Algorithms compared:

ExtraTrees, RandomForest, GradientBoosting, HistGradientBoosting, AdaBoost,
DecisionTree, SVR (RBF), KNN, Ridge, Lasso, ElasticNet, and MLP.

### 6.2 Classification Task

All results are saved to `results/classification results/`.

| File | Description |
|------|-------------|
| `cv5_fold_metrics.xlsx` | Per-fold classification metrics |
| `cv5_metrics_summary.xlsx` | Mean and standard deviation across folds |
| `cv5_classification_report_by_class.xlsx` | Class-level precision, recall, and F1-score |
| `cv5_confusion_matrix_data.xlsx` | Confusion matrix values for each system |
| `cv5_confusion_matrix_<system_name>.png` | Confusion matrix figures |

Classification metrics include Accuracy, Macro-Precision, Macro-Recall,
Macro-F1, and Weighted-F1.

## 7. Methodology Notes

### 7.1 Regression

- 5-fold cross-validation is performed using `KFold` with shuffling
  (`random_state = 42`).
- Interpolation testing uses an 80/20 train–test split
  (`random_state = 42`).
- High-temperature extrapolation uses all samples below the maximum
  temperature for training and the samples at the maximum temperature
  for testing.
- Feature engineering includes temperature, concentration, and derived
  features such as `Temp_C1`, `Inv_T`, `Ln_C1`, `T_Square`, `C1_Square`,
  `Sqrt_T`, and `C1_T_Ratio`.
- Scale-sensitive algorithms (SVR, KNN, Ridge, Lasso, ElasticNet, MLP)
  are wrapped in a `Pipeline` with `StandardScaler`.

### 7.2 Classification

- 5-fold **stratified** cross-validation is used to preserve class
  proportions (`StratifiedKFold`, `random_state = 42`).
- To avoid data leakage, derived features are computed **independently
  within each fold** for the training and test sets. No global statistics
  are used to construct features.
- Solid phases with fewer than 5 samples are excluded before analysis.

## 8. Reproducibility

All random operations use a fixed random seed (`random_state = 42`) to
ensure reproducibility. Running the scripts on the same input data will
produce identical results.

## 9. Contact

For questions about the code or data, please contact the corresponding
author of the manuscript.
liudh@mail.buct.edu.cn 