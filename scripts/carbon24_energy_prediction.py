
"""
Carbon-24 Energy Prediction — Model Leaderboard
=================================================

Pipeline 4 mô hình dự đoán relative_energy:

  Baseline   : Ridge Regression
  Trung cấp  : Random Forest Regressor
  Cao cấp 1  : Gradient Boosting (LightGBM nếu có, fallback sklearn GBM)
  Cao cấp 2  : CatBoost Regressor (tối ưu cho biến categorical)

Input:
  carbon24_pipeline_results/tier3_gmm_labeled.csv
  (đã qua Ground-Truth Labeling — có kmeans_label, gmm_cluster, scientific_label)

Features:
  - 19 structural features (lattice, bond, coordination)
  - Cluster features từ pipeline: kmeans_cluster, gmm_cluster, hdbscan_probability
  - Encoded categorical: crystal_system, space_group_symbol, scientific_label

Target: relative_energy (eV/atom)

Split: dùng cột 'split' có sẵn (train / validation / test)

Output:
  carbon24_energy_results/
  ├── leaderboard.csv
  ├── predictions_test.csv
  ├── feature_importance.csv
  └── figures/
      ├── leaderboard_comparison.png
      ├── predictions_scatter.png
      ├── feature_importance.png
      └── residuals.png
"""

from pathlib import Path
import time
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Optional imports
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

# ============================================================================
# CONFIG
# ============================================================================

RANDOM_STATE = 42
OUTPUT_DIR   = Path("carbon24_energy_results")
FIGURE_DIR   = OUTPUT_DIR / "figures"
DATA_PATH    = Path("carbon24_pipeline_results/tier3_gmm_labeled.csv")

# Structural features (không có energy)
STRUCT_FEATURES = [
    "num_atoms", "a", "b", "c",
    "alpha", "beta", "gamma",
    "volume", "volume_per_atom",
    "b_over_a", "c_over_a", "angle_deviation",
    "mean_bond_length", "std_bond_length",
    "min_bond_length", "max_bond_length",
    "std_coordination", "min_coordination", "max_coordination",
]

# Cluster features từ pipeline (thêm context phân cụm)
CLUSTER_FEATURES = [
    "kmeans_cluster",
    "gmm_cluster",
    "hdbscan_probability",
    "pca1", "pca2",
]

# Categorical features (sẽ được encode)
CAT_FEATURES = [
    "crystal_system",
    "space_group_symbol",
    "scientific_label",
]

TARGET = "relative_energy"


# ============================================================================
# UTILITIES
# ============================================================================

def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")


def regression_metrics(y_true, y_pred, prefix=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    # MAPE: bo qua cac diem co y_true ~ 0 de tranh chia cho 0
    mask = np.abs(y_true) > 0.01
    mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100
            if mask.sum() > 0 else np.nan)
    return {
        f"{prefix}RMSE":  round(rmse, 6),
        f"{prefix}MAE":   round(mae,  6),
        f"{prefix}R2":    round(r2,   6),
        f"{prefix}MAPE%": round(mape, 4) if not np.isnan(mape) else None,
    }


# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_and_prepare():
    print("\n" + "="*65)
    print("DATA PREPARATION")
    print("="*65)

    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded: {df.shape}")

    # Encode categorical features
    le_dict = {}
    for col in CAT_FEATURES:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].fillna("Unknown").astype(str))
            le_dict[col] = le

    # Feature list
    enc_cats = [f"{c}_enc" for c in CAT_FEATURES if c in df.columns]
    all_features = (
        [f for f in STRUCT_FEATURES  if f in df.columns] +
        [f for f in CLUSTER_FEATURES if f in df.columns] +
        enc_cats
    )

    print(f"  Features: {len(all_features)}")
    print(f"    Structural : {len([f for f in STRUCT_FEATURES if f in df.columns])}")
    print(f"    Cluster    : {len([f for f in CLUSTER_FEATURES if f in df.columns])}")
    print(f"    Categorical: {len(enc_cats)}")

    # Split
    train = df[df["split"] == "train"].copy()
    val   = df[df["split"] == "validation"].copy()
    test  = df[df["split"] == "test"].copy()

    print(f"\n  Train : {len(train):,}")
    print(f"  Val   : {len(val):,}")
    print(f"  Test  : {len(test):,}")

    X_train = train[all_features].fillna(train[all_features].median())
    X_val   = val[all_features].fillna(train[all_features].median())
    X_test  = test[all_features].fillna(train[all_features].median())

    y_train = train[TARGET].values
    y_val   = val[TARGET].values
    y_test  = test[TARGET].values

    return X_train, X_val, X_test, y_train, y_val, y_test, all_features, test, le_dict


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_models():
    models = {}

    # ── Baseline: Ridge Regression ──────────────────────────────────────────
    models["Ridge (Baseline)"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])

    # ── Trung cấp: Random Forest ─────────────────────────────────────────────
    models["Random Forest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])

    # ── Cao cấp 1: LightGBM / GBM ────────────────────────────────────────────
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        models["Gradient Boosting"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            )),
        ])

    # ── Cao cấp 2: CatBoost ───────────────────────────────────────────────────
    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            random_seed=RANDOM_STATE,
            verbose=0,
        )
    else:
        # Fallback: GBM tuned
        models["GBM (CatBoost fallback)"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   GradientBoostingRegressor(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=6,
                subsample=0.8,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
            )),
        ])

    return models


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate(models, X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n" + "="*65)
    print("MODEL LEADERBOARD TRAINING")
    print("="*65)

    leaderboard = []
    predictions = {}
    feature_importances = {}

    for name, model in models.items():
        print(f"\n  [{name}]")
        t0 = time.time()

        # Fit
        if HAS_LGB and isinstance(model, lgb.LGBMRegressor):
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(-1)],
            )
        elif HAS_CAT and isinstance(model, CatBoostRegressor):
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        else:
            model.fit(X_train, y_train)

        elapsed = time.time() - t0

        # Predict
        pred_train = model.predict(X_train)
        pred_val   = model.predict(X_val)
        pred_test  = model.predict(X_test)

        # Metrics
        m_train = regression_metrics(y_train, pred_train, "train_")
        m_val   = regression_metrics(y_val,   pred_val,   "val_")
        m_test  = regression_metrics(y_test,  pred_test,  "test_")

        row = {"Model": name, "Train_time_s": round(elapsed, 2)}
        row.update(m_train); row.update(m_val); row.update(m_test)
        leaderboard.append(row)
        predictions[name] = pred_test

        print(f"    Train time : {elapsed:.1f}s")
        print(f"    Val  RMSE  : {m_val['val_RMSE']:.6f}  R2={m_val['val_R2']:.4f}")
        print(f"    Test RMSE  : {m_test['test_RMSE']:.6f}  R2={m_test['test_R2']:.4f}")

        # Feature importance
        try:
            if HAS_LGB and isinstance(model, lgb.LGBMRegressor):
                fi = model.feature_importances_
            elif HAS_CAT and isinstance(model, CatBoostRegressor):
                fi = model.get_feature_importance()
            elif hasattr(model, "named_steps"):
                inner = model.named_steps.get("model", None)
                fi = getattr(inner, "feature_importances_", None)
                if fi is None:
                    fi = getattr(inner, "coef_", None)
            else:
                fi = None
            if fi is not None:
                feature_importances[name] = fi
        except Exception:
            pass

    lb_df = pd.DataFrame(leaderboard).sort_values("test_RMSE")
    return lb_df, predictions, feature_importances


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_leaderboard(lb_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Model Leaderboard — Energy Prediction", fontsize=14, fontweight="bold")

    metrics = [("test_RMSE", "RMSE (lower=better)", "#e74c3c"),
               ("test_R2",   "R² Score (higher=better)", "#2ecc71"),
               ("test_MAE",  "MAE (lower=better)", "#3498db")]

    for ax, (metric, label, color) in zip(axes, metrics):
        vals   = lb_df[metric].values
        models = lb_df["Model"].values
        bars   = ax.barh(models, vals, color=color, alpha=0.8, edgecolor="black")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.5f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(label, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        # Highlight best
        best_idx = vals.argmin() if "RMSE" in metric or "MAE" in metric else vals.argmax()
        axes_bars = ax.patches
        if best_idx < len(axes_bars):
            axes_bars[best_idx].set_edgecolor("gold")
            axes_bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "leaderboard_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  leaderboard_comparison.png")


def plot_predictions_scatter(predictions, y_test, lb_df):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axes = [axes]
    fig.suptitle("Predicted vs Actual Relative Energy (Test Set)",
                 fontsize=14, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for ax, (name, pred), color in zip(axes, predictions.items(), colors):
        r2   = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        ax.scatter(y_test, pred, alpha=0.3, s=8, color=color, edgecolors="none")
        lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        ax.plot(lims, lims, "k--", lw=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual (eV/atom)", fontsize=10)
        ax.set_ylabel("Predicted (eV/atom)", fontsize=10)
        ax.set_title(f"{name}\nR²={r2:.4f}  RMSE={rmse:.5f}", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "predictions_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  predictions_scatter.png")


def plot_feature_importance(feature_importances, feature_names, top_n=20):
    n = len(feature_importances)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(8*n, 8))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Feature Importance (Top {top_n})", fontsize=14, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for ax, (name, fi), color in zip(axes, feature_importances.items(), colors):
        fi_arr = np.array(fi)
        if len(fi_arr) != len(feature_names):
            continue
        fi_series = pd.Series(fi_arr, index=feature_names).abs().sort_values(ascending=False)
        top = fi_series.head(top_n)
        bars = ax.barh(top.index[::-1], top.values[::-1],
                       color=color, alpha=0.8, edgecolor="black")
        ax.set_title(f"{name}", fontweight="bold")
        ax.set_xlabel("Importance", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  feature_importance.png")


def plot_residuals(predictions, y_test):
    n = len(predictions)
    fig, axes = plt.subplots(2, n, figsize=(6*n, 10))
    if n == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Residual Analysis (Test Set)", fontsize=14, fontweight="bold")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for col, (name, pred), color in zip(range(n), predictions.items(), colors):
        residuals = y_test - pred

        # Residual vs predicted
        ax = axes[0, col]
        ax.scatter(pred, residuals, alpha=0.3, s=8, color=color, edgecolors="none")
        ax.axhline(0, color="black", linestyle="--", lw=1.5)
        ax.set_xlabel("Predicted (eV/atom)", fontsize=9)
        ax.set_ylabel("Residual", fontsize=9)
        ax.set_title(f"{name}\nResidual vs Predicted", fontweight="bold", fontsize=10)
        ax.grid(alpha=0.3)

        # Residual distribution
        ax = axes[1, col]
        ax.hist(residuals, bins=60, color=color, alpha=0.7, edgecolor="none", density=True)
        ax.axvline(0, color="black", linestyle="--", lw=1.5)
        ax.axvline(residuals.mean(), color="red", linestyle="-", lw=1.5,
                   label=f"Mean={residuals.mean():.4f}")
        ax.set_xlabel("Residual (eV/atom)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"Residual Distribution\nstd={residuals.std():.4f}", fontweight="bold", fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "residuals.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  residuals.png")


def plot_train_val_test_comparison(lb_df):
    """So sánh RMSE trên train/val/test để phát hiện overfitting."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(lb_df))
    w = 0.25
    colors = ["#3498db", "#f39c12", "#e74c3c"]

    for i, (split, color) in enumerate(zip(["train", "val", "test"], colors)):
        col = f"{split}_RMSE"
        if col in lb_df.columns:
            bars = ax.bar(x + i*w, lb_df[col], w, label=split.capitalize(),
                          color=color, alpha=0.8, edgecolor="black")
            for bar, v in zip(bars, lb_df[col]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                        f"{v:.4f}", ha="center", fontsize=8, rotation=45)

    ax.set_xticks(x + w)
    ax.set_xticklabels(lb_df["Model"], rotation=15, ha="right")
    ax.set_ylabel("RMSE (eV/atom)", fontsize=11)
    ax.set_title("Train / Validation / Test RMSE Comparison\n(Gap = Overfitting indicator)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "train_val_test_rmse.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  train_val_test_rmse.png")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*65)
    print("CARBON-24 ENERGY PREDICTION — MODEL LEADERBOARD")
    print("="*65)
    print(f"  LightGBM : {'available' if HAS_LGB else 'NOT installed (using sklearn GBM)'}")
    print(f"  CatBoost : {'available' if HAS_CAT else 'NOT installed (using GBM fallback)'}")

    ensure_dirs()

    # 1. Data
    X_train, X_val, X_test, y_train, y_val, y_test, features, test_df, le_dict = load_and_prepare()

    # 2. Models
    models = get_models()
    print(f"\n  Models to train: {list(models.keys())}")

    # 3. Train & evaluate
    lb_df, predictions, feature_importances = train_and_evaluate(
        models, X_train, X_val, X_test, y_train, y_val, y_test
    )

    # 4. Print leaderboard
    print("\n" + "="*65)
    print("LEADERBOARD (sorted by test RMSE)")
    print("="*65)
    display_cols = ["Model", "test_RMSE", "test_MAE", "test_R2", "test_MAPE%",
                    "val_RMSE", "val_R2", "Train_time_s"]
    display_cols = [c for c in display_cols if c in lb_df.columns]
    print(lb_df[display_cols].to_string(index=False))

    # 5. Save leaderboard
    lb_df.to_csv(OUTPUT_DIR / "leaderboard.csv", index=False)
    print(f"\n  Saved: leaderboard.csv")

    # 6. Save predictions
    pred_df = test_df[["material_id", "split", "relative_energy",
                        "kmeans_label", "gmm_cluster", "scientific_label",
                        "crystal_system", "space_group_symbol"]].copy()
    for name, pred in predictions.items():
        col = name.replace(" ", "_").replace("(", "").replace(")", "")
        pred_df[f"pred_{col}"] = pred
        pred_df[f"err_{col}"]  = pred_df["relative_energy"] - pred

    # Best model prediction
    best_model = lb_df.iloc[0]["Model"]
    best_col   = best_model.replace(" ", "_").replace("(", "").replace(")", "")
    pred_df["pred_best"] = pred_df[f"pred_{best_col}"]
    pred_df["err_best"]  = pred_df["relative_energy"] - pred_df["pred_best"]

    pred_df.to_csv(OUTPUT_DIR / "predictions_test.csv", index=False)
    print(f"  Saved: predictions_test.csv")

    # 7. Feature importance
    if feature_importances:
        fi_df = pd.DataFrame(feature_importances, index=features)
        fi_df.to_csv(OUTPUT_DIR / "feature_importance.csv")
        print(f"  Saved: feature_importance.csv")

    # 8. Plots
    print("\n  Generating figures...")
    plot_leaderboard(lb_df)
    plot_predictions_scatter(predictions, y_test, lb_df)
    plot_feature_importance(feature_importances, features)
    plot_residuals(predictions, y_test)
    plot_train_val_test_comparison(lb_df)

    # 9. Summary
    best = lb_df.iloc[0]
    print("\n" + "="*65)
    print("BEST MODEL SUMMARY")
    print("="*65)
    print(f"  Model      : {best['Model']}")
    print(f"  Test RMSE  : {best['test_RMSE']:.6f} eV/atom")
    print(f"  Test MAE   : {best['test_MAE']:.6f} eV/atom")
    print(f"  Test R²    : {best['test_R2']:.4f}")
    print(f"  Test MAPE  : {best['test_MAPE%']:.2f}%")
    print(f"\n  Output dir : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
