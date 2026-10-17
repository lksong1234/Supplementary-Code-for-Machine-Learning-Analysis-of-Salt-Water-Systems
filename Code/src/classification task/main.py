# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from collections import Counter

# ====================== Global SCI Times New Roman font configuration ======================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "mathtext.fontset": "dejavuserif",
    "mathtext.rm": "Times New Roman",
    "axes.unicode_minus": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black"
})
EXCEL_FILE = "data.xlsx"
# =============================================================================
# Unified output folder
# =============================================================================
OUTPUT_DIR = "cv5_classification_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
xl = pd.ExcelFile(EXCEL_FILE)
all_sheets = xl.sheet_names
fold_metrics_results = []
cv_summary_results = []
classification_report_results = []
confusion_matrix_dict = {}
for sheet_name in all_sheets:
    print("\n" + "=" * 80)
    print(f"Processing system: {sheet_name}")
    df = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name)
    df.columns = [str(col).strip().replace(" ", "") for col in df.columns]
    # Locate key columns
    col_temp = [c for c in df.columns if '温度' in c or 'Temper' in c][0]
    col_c1 = [c for c in df.columns if '组分1' in c or 'Component1' in c][0]
    col_c2 = [c for c in df.columns if '组分2' in c or 'Component2' in c][0]
    col_solid = [c for c in df.columns if '固相' in c or 'Equilibrium' in c][0]
    # Basic data cleaning: keep only raw base columns and drop missing values
    # (global dataset cleaning, not considered train/test leakage)
    data = df[[col_temp, col_c1, col_c2, col_solid]].copy().dropna()
    data.columns = ["Temperature", "C1", "C2", "Solid"]
    # Global filtering: keep only solid phases with >= 5 samples
    # (dataset preprocessing, not leakage)
    count = Counter(data["Solid"])
    valid = [c for c, v in count.items() if v >= 5]
    data = data[data["Solid"].isin(valid)]
    if len(data) == 0:
        print(f"{sheet_name} has no class satisfying the 5-fold cross-validation requirement, skipping.")
        continue
    # Label encoding
    le = LabelEncoder()
    y_cls = le.fit_transform(data["Solid"])
    n_classes = len(le.classes_)
    if n_classes < 2:
        print(f"{sheet_name} has fewer than 2 valid solid-phase classes, skipping.")
        continue
    # Store only raw base features; do not construct derived features outside the loop (key fix)
    raw_feat_cols = ["Temperature", "C1", "C2"]
    X_raw = data[raw_feat_cols].values
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    y_true_all = []
    y_pred_all = []
    acc_list = []
    precision_macro_list = []
    recall_macro_list = []
    f1_macro_list = []
    f1_weighted_list = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y_cls), 1):
        # Split raw base features
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_cls[train_idx], y_cls[test_idx]
        # ====================== [Leakage-free core] Compute derived features independently within each fold ======================
        # Training set derived features
        T_tr, C1_tr, C2_tr = X_train_raw[:, 0], X_train_raw[:, 1], X_train_raw[:, 2]
        C1_C2_Ratio_tr = C1_tr / (C2_tr + 1e-6)
        Temp_C1_tr = T_tr * C1_tr
        Total_Salt_tr = C1_tr + C2_tr
        X_train = np.column_stack([
            T_tr, C1_tr, C1_C2_Ratio_tr, Temp_C1_tr, Total_Salt_tr
        ])
        # Test set derived features computed only from its own samples,
        # without using global or training-set statistics
        T_te, C1_te, C2_te = X_test_raw[:, 0], X_test_raw[:, 1], X_test_raw[:, 2]
        C1_C2_Ratio_te = C1_te / (C2_te + 1e-6)
        Temp_C1_te = T_te * C1_te
        Total_Salt_te = C1_te + C2_te
        X_test = np.column_stack([
            T_te, C1_te, C1_C2_Ratio_te, Temp_C1_te, Total_Salt_te
        ])
        # =============================================================================
        # System-specific random forest parameters
        if sheet_name == "Na₂SO₄-MgSO₄-H₂O":
            clf = RandomForestClassifier(
                n_estimators=150,
                max_depth=4,
                class_weight="balanced",
                random_state=42
            )
        elif sheet_name == "K₂SO₄-MgSO₄-H₂O":
            clf = RandomForestClassifier(
                n_estimators=300,
                max_depth=3,
                class_weight="balanced_subsample",
                random_state=42
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=150,
                max_depth=4,
                class_weight="balanced",
                random_state=42
            )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # Compute metrics for this fold
        acc = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        acc_list.append(acc)
        precision_macro_list.append(precision_macro)
        recall_macro_list.append(recall_macro)
        f1_macro_list.append(f1_macro)
        f1_weighted_list.append(f1_weighted)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        fold_metrics_results.append({
            "System": sheet_name,
            "Fold": fold,
            "Train_size": len(X_train),
            "Test_size": len(X_test),
            "Num_classes": n_classes,
            "Accuracy": acc,
            "Macro_Precision": precision_macro,
            "Macro_Recall": recall_macro,
            "Macro_F1": f1_macro,
            "Weighted_F1": f1_weighted
        })
        print(
            f"Fold {fold}: "
            f"Acc={acc:.3f}, "
            f"Macro-F1={f1_macro:.3f}, "
            f"Weighted-F1={f1_weighted:.3f}"
        )
    # 5-fold summary metrics
    cv_summary_results.append({
        "System": sheet_name,
        "Total_samples": len(X_raw),
        "Num_classes": n_classes,
        "CV_folds": 5,
        "Accuracy_mean": np.mean(acc_list),
        "Accuracy_std": np.std(acc_list),
        "Macro_Precision_mean": np.mean(precision_macro_list),
        "Macro_Precision_std": np.std(precision_macro_list),
        "Macro_Recall_mean": np.mean(recall_macro_list),
        "Macro_Recall_std": np.std(recall_macro_list),
        "Macro_F1_mean": np.mean(f1_macro_list),
        "Macro_F1_std": np.std(f1_weighted_list),
        "Weighted_F1_mean": np.mean(f1_weighted_list),
        "Weighted_F1_std": np.std(f1_weighted_list)
    })
    print(f"{sheet_name} 5-fold cross-validation completed")
    print(f"Mean Acc: {np.mean(acc_list):.3f} +/- {np.std(acc_list):.3f}")
    print(f"Mean Macro-F1: {np.mean(f1_macro_list):.3f} +/- {np.std(f1_macro_list):.3f}")
    # Confusion matrix
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    cm = confusion_matrix(
        y_true_all,
        y_pred_all,
        labels=np.arange(len(le.classes_))
    )
    cm_df = pd.DataFrame(
        cm,
        index=[f"True_{name}" for name in le.classes_],
        columns=[f"Pred_{name}" for name in le.classes_]
    )
    confusion_matrix_dict[sheet_name] = cm_df
    # Class-level classification report
    report = classification_report(
        y_true_all,
        y_pred_all,
        labels=np.arange(len(le.classes_)),
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )
    for class_name in le.classes_:
        classification_report_results.append({
            "System": sheet_name,
            "Solid_class": class_name,
            "Precision": report[class_name]["precision"],
            "Recall": report[class_name]["recall"],
            "F1-score": report[class_name]["f1-score"],
            "Support": report[class_name]["support"]
        })
    # Plot confusion matrix (font already globally unified, no extra setting needed)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=le.classes_
    )
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d"
    )
    plt.title(f"5-Fold CV Confusion Matrix - {sheet_name}", fontsize=11)
    plt.tight_layout()
    safe_sheet_name = (
        sheet_name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
    )
    fig_path = os.path.join(OUTPUT_DIR, f"cv5_confusion_matrix_{safe_sheet_name}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()
# =============================================================================
# Save all results to the OUTPUT_DIR folder
# =============================================================================
fold_metrics_df = pd.DataFrame(fold_metrics_results)
fold_metrics_df.to_excel(os.path.join(OUTPUT_DIR, "cv5_fold_metrics.xlsx"), index=False)
cv_summary_df = pd.DataFrame(cv_summary_results)
cv_summary_df.to_excel(os.path.join(OUTPUT_DIR, "cv5_metrics_summary.xlsx"), index=False)
classification_report_df = pd.DataFrame(classification_report_results)
classification_report_df.to_excel(os.path.join(OUTPUT_DIR, "cv5_classification_report_by_class.xlsx"), index=False)
with pd.ExcelWriter(os.path.join(OUTPUT_DIR, "cv5_confusion_matrix_data.xlsx")) as writer:
    for sheet_name, cm_df in confusion_matrix_dict.items():
        safe_sheet_name = sheet_name[:31]
        cm_df.to_excel(writer, sheet_name=safe_sheet_name)
print("\nLeakage-free 5-fold cross-validation completed successfully!")
print(f"All results have been saved to folder: {OUTPUT_DIR}")
print(f"Per-fold metrics: {os.path.join(OUTPUT_DIR, 'cv5_fold_metrics.xlsx')}")
print(f"5-fold mean metrics: {os.path.join(OUTPUT_DIR, 'cv5_metrics_summary.xlsx')}")
print(f"Class-level metrics: {os.path.join(OUTPUT_DIR, 'cv5_classification_report_by_class.xlsx')}")
print(f"Confusion matrix data: {os.path.join(OUTPUT_DIR, 'cv5_confusion_matrix_data.xlsx')}")
print("Confusion matrix figures for each system are also saved in the same folder.")