# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:56:55 2026

@author: ls
"""

# -*- coding: utf-8 -*-
"""
Notes:
- algorithms:
  RandomForestRegressor
  GradientBoostingRegressor
  HistGradientBoostingRegressor
  AdaBoostRegressor
  DecisionTreeRegressor
  SVR
  KNeighborsRegressor
  Ridge
  Lasso
  ElasticNet
  MLPRegressor
  ExtraTreesRegressor
"""

import os
import warnings
import logging
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# =============================================================================
# Global settings
# =============================================================================
warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.linewidth"] = 1.2


# =============================================================================
# File path settings
# =============================================================================
EXCEL_FILE = "data.xlsx"

OUTPUT_DIR = "multi_algorithm_regression_results_SCI"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
COMPARE_FIG_DIR = os.path.join(FIG_DIR, "algorithm_comparison")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(COMPARE_FIG_DIR, exist_ok=True)


# =============================================================================
# Cross-validation settings
# =============================================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# =============================================================================
# Multi-algorithm model definitions
# =============================================================================
def get_models():
    """
    Return all regression models to be compared.

    For scale-sensitive models such as SVR, KNN, Ridge, Lasso, ElasticNet,
    and MLP, a Pipeline of StandardScaler + model is used.
    """

    models = {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=31,
            random_state=42
        ),

        "AdaBoost": AdaBoostRegressor(
            n_estimators=300,
            learning_rate=0.03,
            random_state=42
        ),

        "DecisionTree": DecisionTreeRegressor(
            max_depth=6,
            random_state=42
        ),

        "SVR_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(
                kernel="rbf",
                C=50,
                gamma="scale",
                epsilon=0.01
            ))
        ]),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(
                n_neighbors=5,
                weights="distance"
            ))
        ]),

        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(
                alpha=1.0,
                random_state=42
            ))
        ]),

        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(
                alpha=0.001,
                max_iter=10000,
                random_state=42
            ))
        ]),

        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                max_iter=10000,
                random_state=42
            ))
        ]),

        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=3000,
                random_state=42
            ))
        ])
    }

    return models


# =============================================================================
# Utility function: safe filename
# =============================================================================
def safe_filename(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
        .replace(" ", "_")
    )


# =============================================================================
# Utility function: compute evaluation metrics
# =============================================================================
def calc_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return r2, rmse, mae


# =============================================================================
# SCI-style experimental vs. predicted scatter plot
# =============================================================================
def plot_sci_prediction_scatter(
    y_train,
    pred_train,
    y_test,
    pred_test,
    sheet_name,
    model_name,
    save_path,
    title_text
):
    y_train = np.asarray(y_train)
    pred_train = np.asarray(pred_train)
    y_test = np.asarray(y_test)
    pred_test = np.asarray(pred_test)

    y_all = np.concatenate([y_train, y_test])
    pred_all = np.concatenate([pred_train, pred_test])

    r2_test, rmse_test, mae_test = calc_regression_metrics(y_test, pred_test)

    min_val = min(np.min(y_all), np.min(pred_all))
    max_val = max(np.max(y_all), np.max(pred_all))

    padding = 0.08 * (max_val - min_val) if max_val > min_val else 1.0
    axis_min = min_val - padding
    axis_max = max_val + padding

    fit_x = np.linspace(axis_min, axis_max, 200)

    if len(y_train) >= 2:
        coef_train = np.polyfit(y_train, pred_train, 1)
        fit_train = np.polyval(coef_train, fit_x)
    else:
        fit_train = None

    if len(y_test) >= 2:
        coef_test = np.polyfit(y_test, pred_test, 1)
        fit_test = np.polyval(coef_test, fit_x)
    else:
        fit_test = None

    fig, ax = plt.subplots(figsize=(4.8, 4.4), dpi=300)

    ax.scatter(
        y_train,
        pred_train,
        s=34,
        marker="o",
        facecolors="#4C78A8",
        edgecolors="white",
        linewidths=0.45,
        alpha=0.78,
        label="Train"
    )

    ax.scatter(
        y_test,
        pred_test,
        s=40,
        marker="s",
        facecolors="#F58518",
        edgecolors="white",
        linewidths=0.45,
        alpha=0.85,
        label="Test"
    )

    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        linestyle="--",
        color="black",
        linewidth=1.2,
        alpha=0.75,
        label="1:1 line"
    )

    if fit_train is not None:
        ax.plot(
            fit_x,
            fit_train,
            color="#4C78A8",
            linewidth=1.8,
            label="Train fit"
        )

    if fit_test is not None:
        ax.plot(
            fit_x,
            fit_test,
            color="#F58518",
            linewidth=1.8,
            label="Test fit"
        )

    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)

    ax.set_xlabel("Experimental $C_2$", fontsize=12)
    ax.set_ylabel("Predicted $C_2$", fontsize=12)

    ax.set_title(title_text, fontsize=11.5, pad=8)

    metric_text = (
        f"{model_name}\n"
        f"$R^2$ = {r2_test:.3f}\n"
        f"RMSE = {rmse_test:.3f}\n"
        f"MAE = {mae_test:.3f}"
    )

    ax.text(
        0.05,
        0.95,
        metric_text,
        transform=ax.transAxes,
        fontsize=9.5,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="gray",
            linewidth=0.8,
            alpha=0.92
        )
    )

    ax.legend(
        loc="lower right",
        fontsize=8.2,
        frameon=True,
        framealpha=0.95,
        edgecolor="gray"
    )

    ax.tick_params(
        direction="in",
        length=4,
        width=1.0,
        labelsize=10,
        top=True,
        right=True
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.35
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# Algorithm performance comparison bar chart
# =============================================================================
def plot_algorithm_comparison_bar(
    df,
    system_name,
    mode_name,
    metric_name,
    save_path
):
    if df.empty:
        return

    plot_df = df[df["System"] == system_name].copy()

    if plot_df.empty:
        return

    if metric_name not in plot_df.columns:
        return

    plot_df = plot_df.sort_values(metric_name, ascending=(metric_name != "R2"))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)

    ax.bar(
        plot_df["Model"],
        plot_df[metric_name],
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85
    )

    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_title(f"{mode_name} {metric_name} comparison - {system_name}", fontsize=12)

    ax.tick_params(
        axis="x",
        rotation=45,
        labelsize=9,
        direction="in"
    )

    ax.tick_params(
        axis="y",
        labelsize=10,
        direction="in",
        right=True
    )

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# Feature engineering function
# =============================================================================
def build_features(data):
    T = data["Temperature"].astype(float).clip(1e-6, None)
    C1 = data["C1"].astype(float).clip(1e-6, None)

    data["Temp_C1"] = T * C1
    data["Inv_T"] = 1.0 / T
    data["Ln_C1"] = np.log(C1)
    data["T_Square"] = T ** 2
    data["C1_Square"] = C1 ** 2
    data["Sqrt_T"] = np.sqrt(T)
    data["C1_T_Ratio"] = C1 / T

    feats = [
        "Temperature",
        "C1",
        "Temp_C1",
        "Inv_T",
        "Ln_C1",
        "T_Square",
        "C1_Square",
        "Sqrt_T",
        "C1_T_Ratio"
    ]

    data = data.dropna(subset=feats)

    return data, feats


# =============================================================================
# Main program
# =============================================================================
xl = pd.ExcelFile(EXCEL_FILE)
all_sheets = xl.sheet_names

cv_results = []
cv_summary_results = []

interpolation_metrics_results = []
extrapolation_metrics_results = []

interpolation_prediction_dict = {}
extrapolation_prediction_dict = {}

models = get_models()

for sheet_name in all_sheets:
    print("\n" + "=" * 90)
    print(f"Processing system: {sheet_name}")

    safe_system_name = safe_filename(sheet_name)

    system_fig_dir = os.path.join(FIG_DIR, safe_system_name)
    os.makedirs(system_fig_dir, exist_ok=True)

    # =========================================================================
    # Read data
    # =========================================================================
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
    df.columns = [str(col).strip().replace(" ", "") for col in df.columns]

    try:
        col_temp = [c for c in df.columns if "温度" in c or "Temper" in c][0]
        col_c1 = [c for c in df.columns if "组分1" in c or "Component1" in c][0]
        col_c2 = [c for c in df.columns if "组分2" in c or "Component2" in c][0]
        col_solid = [c for c in df.columns if "固相" in c or "Solid" in c][0]

    except IndexError:
        print(f"Skipping {sheet_name}: required columns not found.")
        continue

    data = df[[col_temp, col_c1, col_c2, col_solid]].copy().dropna()
    data.columns = ["Temperature", "C1", "C2", "Solid"]

    # =========================================================================
    # Feature engineering
    # =========================================================================
    data, feats = build_features(data)

    # Filter solid-phase classes with too few samples
    data_all = data.copy()
    cnt = Counter(data_all["Solid"])
    valid_cls = [k for k, v in cnt.items() if v >= 2]
    data_all = data_all[data_all["Solid"].isin(valid_cls)]

    X = data_all[feats].values
    y_reg = data_all["C2"].values

    if len(X) < 10:
        print(f"{sheet_name} has too few valid samples, skipping.")
        continue

    print(f"Number of valid samples: {len(X)}")
    print(f"Number of features: {len(feats)}")
    print(f"Number of algorithms compared: {len(models)}")

    # =========================================================================
    # Evaluate each model
    # =========================================================================
    for model_name, model in models.items():

        print("\n" + "-" * 70)
        print(f"Current algorithm: {model_name}")

        safe_model_name = safe_filename(model_name)

        # =====================================================================
        # 1. 5-fold cross-validation
        # =====================================================================
        cv_r2 = []
        cv_rmse = []
        cv_mae = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):

            Xtr, Xval = X[tr_idx], X[val_idx]
            ytr, yval = y_reg[tr_idx], y_reg[val_idx]

            try:
                model.fit(Xtr, ytr)
                pred_val = model.predict(Xval)

                r2_fold, rmse_fold, mae_fold = calc_regression_metrics(
                    yval,
                    pred_val
                )

            except Exception as e:
                print(f"{model_name} Fold {fold} error: {e}")
                r2_fold, rmse_fold, mae_fold = np.nan, np.nan, np.nan

            cv_r2.append(r2_fold)
            cv_rmse.append(rmse_fold)
            cv_mae.append(mae_fold)

            cv_results.append({
                "System": sheet_name,
                "Model": model_name,
                "Fold": fold,
                "R2": r2_fold,
                "RMSE": rmse_fold,
                "MAE": mae_fold,
                "Train_size": len(Xtr),
                "Validation_size": len(Xval)
            })

        mean_r2 = np.nanmean(cv_r2)
        std_r2 = np.nanstd(cv_r2)
        mean_rmse = np.nanmean(cv_rmse)
        std_rmse = np.nanstd(cv_rmse)
        mean_mae = np.nanmean(cv_mae)
        std_mae = np.nanstd(cv_mae)

        cv_summary_results.append({
            "System": sheet_name,
            "Model": model_name,
            "R2_mean": mean_r2,
            "R2_std": std_r2,
            "RMSE_mean": mean_rmse,
            "RMSE_std": std_rmse,
            "MAE_mean": mean_mae,
            "MAE_std": std_mae
        })

        print(
            f"5-fold CV: R2 = {mean_r2:.3f} +/- {std_r2:.3f}, "
            f"RMSE = {mean_rmse:.3f}, MAE = {mean_mae:.3f}"
        )

        # =====================================================================
        # 2. Interpolation test
        # =====================================================================
        Xtr, Xte, ytr, yte = train_test_split(
            X,
            y_reg,
            test_size=0.2,
            random_state=42
        )

        try:
            model.fit(Xtr, ytr)

            pred_tr = model.predict(Xtr)
            pred_te = model.predict(Xte)

            r2_te, rmse_te, mae_te = calc_regression_metrics(yte, pred_te)

            print(
                f"Interpolation test: R2 = {r2_te:.3f}, "
                f"RMSE = {rmse_te:.3f}, MAE = {mae_te:.3f}"
            )

            interpolation_metrics_results.append({
                "System": sheet_name,
                "Model": model_name,
                "Mode": "Interpolation",
                "Train_size": len(Xtr),
                "Test_size": len(Xte),
                "R2": r2_te,
                "RMSE": rmse_te,
                "MAE": mae_te
            })

            interp_pred_df = pd.DataFrame({
                "System": sheet_name,
                "Model": model_name,
                "Mode": "Interpolation",
                "Dataset": ["Train"] * len(ytr) + ["Test"] * len(yte),
                "True_C2": np.concatenate([ytr, yte]),
                "Predicted_C2": np.concatenate([pred_tr, pred_te]),
                "Residual": np.concatenate([pred_tr - ytr, pred_te - yte])
            })

            interp_key = f"{sheet_name}_{model_name}"
            interpolation_prediction_dict[interp_key] = interp_pred_df

            interp_fig_path = os.path.join(
                system_fig_dir,
                f"{safe_system_name}_{safe_model_name}_interpolation_scatter.png"
            )

            plot_sci_prediction_scatter(
                y_train=ytr,
                pred_train=pred_tr,
                y_test=yte,
                pred_test=pred_te,
                sheet_name=sheet_name,
                model_name=model_name,
                save_path=interp_fig_path,
                title_text=f"Interpolation - {sheet_name}"
            )

        except Exception as e:
            print(f"{model_name} interpolation test error: {e}")

        # =====================================================================
        # 3. High-temperature extrapolation test
        # =====================================================================
        maxT = data_all["Temperature"].max()

        train_ext = data_all[data_all["Temperature"] < maxT].dropna()
        test_ext = data_all[data_all["Temperature"] == maxT].dropna()

        if len(train_ext) > 0 and len(test_ext) > 0:

            Xtr_ext = train_ext[feats].values
            ytr_ext = train_ext["C2"].values

            Xte_ext = test_ext[feats].values
            yte_ext = test_ext["C2"].values

            try:
                model.fit(Xtr_ext, ytr_ext)

                pred_tr_ext = model.predict(Xtr_ext)
                pred_te_ext = model.predict(Xte_ext)

                r2_ext, rmse_ext, mae_ext = calc_regression_metrics(
                    yte_ext,
                    pred_te_ext
                )

                print(
                    f"High-temperature extrapolation: R2 = {r2_ext:.3f}, "
                    f"RMSE = {rmse_ext:.3f}, MAE = {mae_ext:.3f}"
                )

                extrapolation_metrics_results.append({
                    "System": sheet_name,
                    "Model": model_name,
                    "Mode": "High-temperature extrapolation",
                    "Train_size": len(Xtr_ext),
                    "Test_size": len(Xte_ext),
                    "Extrapolated_temperature": maxT,
                    "R2": r2_ext,
                    "RMSE": rmse_ext,
                    "MAE": mae_ext
                })

                ext_pred_df = pd.DataFrame({
                    "System": sheet_name,
                    "Model": model_name,
                    "Mode": "High-temperature extrapolation",
                    "Dataset": ["Train"] * len(ytr_ext) + ["Test"] * len(yte_ext),
                    "Temperature": np.concatenate([
                        train_ext["Temperature"].values,
                        test_ext["Temperature"].values
                    ]),
                    "True_C2": np.concatenate([ytr_ext, yte_ext]),
                    "Predicted_C2": np.concatenate([pred_tr_ext, pred_te_ext]),
                    "Residual": np.concatenate([
                        pred_tr_ext - ytr_ext,
                        pred_te_ext - yte_ext
                    ])
                })

                ext_key = f"{sheet_name}_{model_name}"
                extrapolation_prediction_dict[ext_key] = ext_pred_df

                ext_fig_path = os.path.join(
                    system_fig_dir,
                    f"{safe_system_name}_{safe_model_name}_extrapolation_scatter.png"
                )

                plot_sci_prediction_scatter(
                    y_train=ytr_ext,
                    pred_train=pred_tr_ext,
                    y_test=yte_ext,
                    pred_test=pred_te_ext,
                    sheet_name=sheet_name,
                    model_name=model_name,
                    save_path=ext_fig_path,
                    title_text=f"High-temperature extrapolation - {sheet_name}"
                )

            except Exception as e:
                print(f"{model_name} high-temperature extrapolation error: {e}")

        else:
            print(f"{model_name} cannot perform high-temperature extrapolation: training or extrapolation test set is empty.")


# =============================================================================
# Save 5-fold cross-validation results
# =============================================================================
cv_results_df = pd.DataFrame(cv_results)
cv_results_df.to_excel(
    os.path.join(DATA_DIR, "cv5_metrics_by_fold_all_models.xlsx"),
    index=False
)

cv_summary_df = pd.DataFrame(cv_summary_results)
cv_summary_df.to_excel(
    os.path.join(DATA_DIR, "cv5_metrics_summary_all_models.xlsx"),
    index=False
)


# =============================================================================
# Save interpolation results
# =============================================================================
interpolation_metrics_df = pd.DataFrame(interpolation_metrics_results)
interpolation_metrics_df.to_excel(
    os.path.join(DATA_DIR, "interpolation_metrics_all_models.xlsx"),
    index=False
)


# =============================================================================
# Save high-temperature extrapolation results
# =============================================================================
extrapolation_metrics_df = pd.DataFrame(extrapolation_metrics_results)
extrapolation_metrics_df.to_excel(
    os.path.join(DATA_DIR, "extrapolation_metrics_all_models.xlsx"),
    index=False
)


# =============================================================================
# Save interpolation prediction data
# =============================================================================
with pd.ExcelWriter(
    os.path.join(DATA_DIR, "interpolation_prediction_data_all_models.xlsx")
) as writer:

    for key, pred_df in interpolation_prediction_dict.items():
        safe_sheet = safe_filename(key)[:31]
        pred_df.to_excel(writer, sheet_name=safe_sheet, index=False)


# =============================================================================
# Save high-temperature extrapolation prediction data
# =============================================================================
with pd.ExcelWriter(
    os.path.join(DATA_DIR, "extrapolation_prediction_data_all_models.xlsx")
) as writer:

    for key, pred_df in extrapolation_prediction_dict.items():
        safe_sheet = safe_filename(key)[:31]
        pred_df.to_excel(writer, sheet_name=safe_sheet, index=False)


# =============================================================================
# Save best model summary
# =============================================================================
best_model_results = []

if not cv_summary_df.empty:
    for system_name in cv_summary_df["System"].unique():
        tmp = cv_summary_df[cv_summary_df["System"] == system_name].copy()

        tmp_r2 = tmp.dropna(subset=["R2_mean"])
        tmp_rmse = tmp.dropna(subset=["RMSE_mean"])
        tmp_mae = tmp.dropna(subset=["MAE_mean"])

        if not tmp_r2.empty:
            best_r2_row = tmp_r2.loc[tmp_r2["R2_mean"].idxmax()]
        else:
            best_r2_row = None

        if not tmp_rmse.empty:
            best_rmse_row = tmp_rmse.loc[tmp_rmse["RMSE_mean"].idxmin()]
        else:
            best_rmse_row = None

        if not tmp_mae.empty:
            best_mae_row = tmp_mae.loc[tmp_mae["MAE_mean"].idxmin()]
        else:
            best_mae_row = None

        best_model_results.append({
            "System": system_name,

            "Best_CV_R2_Model": best_r2_row["Model"] if best_r2_row is not None else np.nan,
            "Best_CV_R2": best_r2_row["R2_mean"] if best_r2_row is not None else np.nan,

            "Best_CV_RMSE_Model": best_rmse_row["Model"] if best_rmse_row is not None else np.nan,
            "Best_CV_RMSE": best_rmse_row["RMSE_mean"] if best_rmse_row is not None else np.nan,

            "Best_CV_MAE_Model": best_mae_row["Model"] if best_mae_row is not None else np.nan,
            "Best_CV_MAE": best_mae_row["MAE_mean"] if best_mae_row is not None else np.nan
        })

best_model_df = pd.DataFrame(best_model_results)
best_model_df.to_excel(
    os.path.join(DATA_DIR, "best_model_summary_by_cv.xlsx"),
    index=False
)


# =============================================================================
# Plot algorithm comparison charts
# =============================================================================
if not interpolation_metrics_df.empty:
    for system_name in interpolation_metrics_df["System"].unique():

        safe_system_name = safe_filename(system_name)

        for metric in ["R2", "RMSE", "MAE"]:
            save_path = os.path.join(
                COMPARE_FIG_DIR,
                f"{safe_system_name}_interpolation_{metric}_comparison.png"
            )

            plot_algorithm_comparison_bar(
                df=interpolation_metrics_df,
                system_name=system_name,
                mode_name="Interpolation",
                metric_name=metric,
                save_path=save_path
            )

if not extrapolation_metrics_df.empty:
    for system_name in extrapolation_metrics_df["System"].unique():

        safe_system_name = safe_filename(system_name)

        for metric in ["R2", "RMSE", "MAE"]:
            save_path = os.path.join(
                COMPARE_FIG_DIR,
                f"{safe_system_name}_extrapolation_{metric}_comparison.png"
            )

            plot_algorithm_comparison_bar(
                df=extrapolation_metrics_df,
                system_name=system_name,
                mode_name="High-temperature extrapolation",
                metric_name=metric,
                save_path=save_path
            )


# =============================================================================
# Terminal output summary
# =============================================================================
print("\n" + "=" * 90)
print("Multi-algorithm regression comparison completed successfully!")
print(f"All results have been saved to folder: {OUTPUT_DIR}")
print(f"Figures saved to: {FIG_DIR}")
print(f"Algorithm comparison figures saved to: {COMPARE_FIG_DIR}")
print(f"Data saved to: {DATA_DIR}")

print("\nSaved files include:")
print("1. cv5_metrics_by_fold_all_models.xlsx")
print("2. cv5_metrics_summary_all_models.xlsx")
print("3. interpolation_metrics_all_models.xlsx")
print("4. extrapolation_metrics_all_models.xlsx")
print("5. interpolation_prediction_data_all_models.xlsx")
print("6. extrapolation_prediction_data_all_models.xlsx")
print("7. best_model_summary_by_cv.xlsx")

print("\nRecommended priorities for manuscript:")
print("1. cv5_metrics_summary_all_models.xlsx: overall generalization ability")
print("2. interpolation_metrics_all_models.xlsx: random-split interpolation prediction ability")
print("3. extrapolation_metrics_all_models.xlsx: high-temperature extrapolation prediction ability")
print("4. best_model_summary_by_cv.xlsx: best model for each system")