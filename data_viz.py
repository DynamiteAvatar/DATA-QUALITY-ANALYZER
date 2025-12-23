import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------- UTIL ----------------
def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


# ============================================================
# 1. MISSING DATA HEATMAP
# ============================================================
def generate_missing_data_heatmap(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.heatmap(
        df.isnull(),
        cmap="viridis",
        cbar=False,
        ax=ax
    )

    ax.set_title("Missing Data Structure", fontsize=11)
    ax.set_xlabel("Columns", fontsize=9)
    ax.set_ylabel("Rows", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    return _fig_to_base64(fig)


# ============================================================
# 2. CORRELATION MATRIX
# ============================================================
def generate_feature_correlation_clustermap(df: pd.DataFrame) -> str:
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "Not enough numeric columns",
                ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _fig_to_base64(fig)

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar=True,
        ax=ax,
        annot_kws={"size": 8}
    )

    ax.set_title("Feature Correlation Matrix", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    return _fig_to_base64(fig)


# ============================================================
# 3. NUMERICAL DISTRIBUTION (SKEW VIEW – YOUR PREFERRED STYLE)
# ============================================================
def generate_numerical_distribution_plot(df: pd.DataFrame) -> str:
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "No numeric columns",
                ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _fig_to_base64(fig)

    fig, axes = plt.subplots(
        nrows=len(numeric_df.columns),
        ncols=1,
        figsize=(6, 2.5 * len(numeric_df.columns))
    )

    if len(numeric_df.columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_df.columns):
        data = numeric_df[col].dropna()
        skew = round(data.skew(), 2)

        sns.kdeplot(data, ax=ax, fill=True, color="#4F46E5", alpha=0.6)
        ax.axvline(data.mean(), color="black", linestyle="--", linewidth=1)

        ax.set_title(f"{col} | Skewness: {skew}", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    return _fig_to_base64(fig)


# ============================================================
# 4. COLUMN QUALITY RISK MATRIX
# ============================================================
def generate_column_quality_heatmap(df: pd.DataFrame) -> str:
    metrics = []

    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        uniqueness = df[col].nunique() / max(len(df), 1) * 100

        metrics.append({
            "Missing %": round(missing_pct, 2),
            "Uniqueness %": round(uniqueness, 2),
        })

    quality_df = pd.DataFrame(metrics, index=df.columns)

    fig, ax = plt.subplots(figsize=(6, max(3, len(df.columns) * 0.3)))

    sns.heatmap(
        quality_df,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.4,
        cbar=True,
        ax=ax,
        annot_kws={"size": 8}
    )

    ax.set_title("Column Quality Risk Matrix", fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    return _fig_to_base64(fig)
