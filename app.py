from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage

from ml_engine import (
    CLASSIFIERS, REGRESSORS, DATASET_PATHS, TARGETS,
    load_dataset, evaluate_classifier, evaluate_regressor, run_clustering,
)

# -----------------------------------------------------------------------------
# Page setup and shared styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bias Breakers | ML Analytics",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp { background: linear-gradient(145deg, #07111f 0%, #0b1728 52%, #101b31 100%); }
      [data-testid="stHeader"] { background: rgba(7,17,31,0.86); }
      [data-testid="stSidebar"] { background: linear-gradient(180deg,#081322 0%,#0d1b2e 100%); border-right: 1px solid #22334d; }
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #e5f0ff; }
      h1, h2, h3 { color: #e8f1ff; letter-spacing: -0.025em; }
      p, li, label, [data-testid="stMarkdownContainer"] { color: #c5d2e6; }
      .hero { padding: 1.35rem 1.5rem; border-radius: 18px; border: 1px solid #263b59;
              background: linear-gradient(120deg, rgba(25,58,92,.88), rgba(13,30,53,.93)); margin-bottom: 1rem; }
      .hero-kicker { color: #7dd3fc; font-size: .82rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase; }
      .hero-title { color: #f2f7ff; font-size: clamp(1.8rem,3vw,2.7rem); font-weight: 800; margin: .25rem 0 .3rem; }
      .hero-subtitle { color: #b9cbe0; font-size: 1rem; margin-bottom: 0; }
      [data-testid="stMetric"] { background: rgba(17,35,58,.78); border: 1px solid #263b59; padding: 1rem 1.1rem; border-radius: 14px; }
      [data-testid="stMetricLabel"] { color: #a9bfd8; }
      [data-testid="stMetricValue"] { color: #eff7ff; }
      [data-testid="stDataFrame"] { border: 1px solid #273a54; border-radius: 12px; }
      .section-note { color: #91a9c4; font-size: .9rem; }
      div.stButton > button { border-radius: 10px; border: 1px solid #3d83bd; font-weight: 650; min-height: 2.7rem; }
      div.stButton > button[kind="primary"] { background: linear-gradient(90deg,#087ca8,#2563a6); color: white; }
      [data-testid="stAlert"] { border-radius: 12px; }
      footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit checks a very small file signature on each rerun; content loading stays cached.
def _dataset_stamp(dataset_key: str) -> str:
    p = DATASET_PATHS[dataset_key]
    try:
        stat = p.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        return "missing"


@st.cache_data(show_spinner=False)
def cached_dataset(dataset_key: str, stamp: str) -> pd.DataFrame:
    # stamp deliberately participates in the cache key so replacing a CSV refreshes it.
    return load_dataset(dataset_key)


@st.cache_data(show_spinner=False, max_entries=40)
def cached_classification(dataset_key: str, model_name: str, excluded_features: tuple[str, ...], stamp: str):
    return evaluate_classifier(dataset_key, model_name, excluded_features)


@st.cache_data(show_spinner=False, max_entries=30)
def cached_regression(dataset_key: str, model_name: str, excluded_features: tuple[str, ...], stamp: str):
    return evaluate_regressor(dataset_key, model_name, excluded_features)


@st.cache_data(show_spinner=False, max_entries=10)
def cached_clustering(n_clusters: int, include_tsne: bool, stamp: str):
    return run_clustering(n_clusters=n_clusters, seed=42, include_tsne=include_tsne)


def fmt_num(value, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def header(title: str, subtitle: str, eyebrow: str = "BIAS BREAKERS • ML ANALYTICS") -> None:
    st.markdown(
        f'<div class="hero"><div class="hero-kicker">{eyebrow}</div>'
        f'<div class="hero-title">{title}</div><p class="hero-subtitle">{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def plotly_chart(fig, key: str | None = None) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,20,35,0.32)",
        font=dict(color="#d9e6f7"),
        margin=dict(l=16, r=16, t=54, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)


def display_dataset_key() -> str:
    return st.selectbox("Choose dataset", list(DATASET_PATHS.keys()), key="dataset_global_select")


def safe_load(dataset_key: str) -> pd.DataFrame | None:
    stamp = _dataset_stamp(dataset_key)
    try:
        with st.spinner(f"Loading {Path(DATASET_PATHS[dataset_key]).name}…"):
            return cached_dataset(dataset_key, stamp)
    except FileNotFoundError as exc:
        st.error(str(exc))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"Could not read the dataset: {exc}")
    return None


def show_quality_alerts(frame: pd.DataFrame, include_duplicate_note: bool = True) -> None:
    missing = int(frame.isna().sum().sum())
    duplicate = int(frame.duplicated().sum())
    cols = st.columns(3)
    cols[0].metric("Missing cells", f"{missing:,}")
    cols[1].metric("Duplicate rows", f"{duplicate:,}")
    cols[2].metric("Memory use", f"{frame.memory_usage(deep=True).sum() / 1024:.1f} KB")
    if missing:
        st.info(f"This dataset contains {missing:,} missing cells. Model pipelines impute predictor values on the training split; rows with missing supervised targets are excluded from that evaluation.")
    if duplicate and include_duplicate_note:
        st.warning(f"Found {duplicate:,} exact duplicate rows. The classification evaluation removes exact duplicates before splitting to reduce train/test overlap risk; the explorer shows the original dataset unchanged.")


def render_overview() -> None:
    header("Bias Breakers", "A single workspace to explore data, compare ML models, and inspect clustering results.")
    st.caption("This UI is built for the datasets included in the uploaded ML Capstone project: diabetes classification, insurance-cost regression, and patient clustering.")

    datasets = {}
    errors = []
    for key in DATASET_PATHS:
        try:
            datasets[key] = cached_dataset(key, _dataset_stamp(key))
        except Exception as exc:
            errors.append(f"{key}: {exc}")
    if errors:
        for error in errors:
            st.warning(error)
    if not datasets:
        st.error("No project datasets could be loaded. Check that the dataset/ folder is beside app.py.")
        return

    total_rows = sum(len(d) for d in datasets.values())
    total_features = sum(d.shape[1] for d in datasets.values())
    total_missing = sum(int(d.isna().sum().sum()) for d in datasets.values())
    data_size = sum(d.memory_usage(deep=True).sum() for d in datasets.values()) / (1024 * 1024)
    a, b, c, d = st.columns(4)
    a.metric("Datasets available", len(datasets))
    b.metric("Total records", f"{total_rows:,}")
    c.metric("Missing cells", f"{total_missing:,}")
    d.metric("In-memory data size", f"{data_size:.2f} MB")

    left, right = st.columns([1.12, 1])
    with left:
        st.subheader("Project datasets")
        summary_rows = []
        for key, frame in datasets.items():
            target = TARGETS.get(key, "—")
            summary_rows.append({
                "Dataset": key,
                "Rows": len(frame),
                "Columns": frame.shape[1],
                "Missing cells": int(frame.isna().sum().sum()),
                "Target / role": target if key != "Patient clustering" else "Unsupervised",
            })
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Data completeness")
        completeness = pd.DataFrame([
            {"Dataset": k, "Completeness (%)": 100 * (1 - d.isna().sum().sum() / max(1, d.size))}
            for k, d in datasets.items()
        ])
        fig = px.bar(completeness, x="Dataset", y="Completeness (%)", text="Completeness (%)", range_y=[0, 105])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        plotly_chart(fig, "overview_completeness")

    st.subheader("Quick look at the supervised targets")
    tab1, tab2 = st.tabs(["Diabetes target", "Insurance charges"])
    with tab1:
        diabetes = datasets.get("Diabetes classification")
        if diabetes is not None and "diabetes" in diabetes:
            counts = diabetes["diabetes"].value_counts(dropna=False).sort_index().rename_axis("Diabetes label").reset_index(name="Records")
            counts["Class"] = counts["Diabetes label"].map({0: "No diabetes (0)", 1: "Diabetes (1)"}).fillna(counts["Diabetes label"].astype(str))
            fig = px.bar(counts, x="Class", y="Records", text="Records", color="Class", title="Diabetes class distribution")
            plotly_chart(fig, "overview_diabetes_target")
            st.caption("Class distribution is shown from the original CSV. The evaluation workflow reports the held-out test metrics separately.")
    with tab2:
        insurance = datasets.get("Insurance regression")
        if insurance is not None and "charges" in insurance:
            valid = insurance["charges"].dropna()
            fig = px.histogram(valid.to_frame(), x="charges", nbins=35, title="Insurance charge distribution", labels={"charges": "Insurance charges"})
            plotly_chart(fig, "overview_charges")

    st.subheader("Existing notebook results")
    project_root = Path(__file__).resolve().parent
    cls_path = project_root / "results" / "classification_results.csv"
    clu_path = project_root / "results" / "clustering_comparison.csv"
    result_left, result_right = st.columns(2)
    with result_left:
        if cls_path.exists():
            try:
                result_df = pd.read_csv(cls_path)
                st.markdown("**Saved classification comparison**")
                st.dataframe(result_df, hide_index=True, use_container_width=True)
            except Exception as exc:
                st.caption(f"Saved classification results could not be read: {exc}")
        else:
            st.caption("No saved classification comparison found yet. Run a comparison from the Classification page.")
    with result_right:
        if clu_path.exists():
            try:
                st.markdown("**Saved clustering comparison**")
                st.dataframe(pd.read_csv(clu_path), hide_index=True, use_container_width=True)
            except Exception as exc:
                st.caption(f"Saved clustering results could not be read: {exc}")
        else:
            st.caption("No saved clustering comparison found yet. Run the Clustering page.")


def _missing_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"Feature": frame.columns, "Missing count": frame.isna().sum().values})
    out["Missing (%)"] = (100 * out["Missing count"] / max(1, len(frame))).round(2)
    out["Data type"] = frame.dtypes.astype(str).values
    out["Unique values"] = [frame[c].nunique(dropna=True) for c in frame.columns]
    return out.sort_values(["Missing count", "Feature"], ascending=[False, True])


def render_all_features(frame: pd.DataFrame, dataset_key: str) -> None:
    numeric = frame.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    qtab, ntab, ctab, reltab, summtab = st.tabs(["Data quality", "Numeric features", "Categorical features", "Relationships", "Summary tables"])

    with qtab:
        missing_df = _missing_frame(frame)
        q1, q2 = st.columns(2)
        with q1:
            st.markdown("**Missing values by feature**")
            missing_nonzero = missing_df[missing_df["Missing count"] > 0]
            plot_df = missing_nonzero if not missing_nonzero.empty else missing_df.head(15)
            fig = px.bar(plot_df, x="Missing count", y="Feature", orientation="h", color="Missing (%)", title="Missing-value counts")
            plotly_chart(fig, f"all_missing_{dataset_key}")
        with q2:
            st.markdown("**Feature types and cardinality**")
            kind_df = missing_df.copy()
            kind_df["Type group"] = np.where(kind_df["Feature"].isin(numeric), "Numeric", "Categorical")
            fig = px.bar(kind_df, x="Feature", y="Unique values", color="Type group", title="Unique values per feature")
            fig.update_layout(xaxis_tickangle=-35)
            plotly_chart(fig, f"all_unique_{dataset_key}")
        show_quality_alerts(frame)
        st.dataframe(missing_df, hide_index=True, use_container_width=True)
        st.markdown("**Feature-quality notes**")
        constant_cols = [c for c in frame.columns if frame[c].nunique(dropna=False) <= 1]
        all_missing = [c for c in frame.columns if frame[c].isna().all()]
        st.write({
            "all_missing_columns": all_missing,
            "constant_or_single_value_columns": constant_cols,
            "duplicate_rows": int(frame.duplicated().sum()),
            "rows_with_any_missing": int(frame.isna().any(axis=1).sum()),
        })

    with ntab:
        if not numeric:
            st.info("This dataset has no numeric columns.")
        else:
            for feature in numeric[:6]:
                with st.expander(f"{feature} — distribution", expanded=(numeric.index(feature) == 0)):
                    f1, f2 = st.columns([1.4, 1])
                    with f1:
                        fig = px.histogram(frame, x=feature, nbins=35, marginal="box", title=f"Distribution of {feature}")
                        plotly_chart(fig, f"all_num_dist_{dataset_key}_{feature}")
                    with f2:
                        st.dataframe(frame[feature].describe().to_frame("Value"), use_container_width=True)
            if len(numeric) > 6:
                st.caption("To keep the page responsive, the first six numeric features are expanded here. Use the individual feature selector to inspect any remaining feature.")

    with ctab:
        if not categorical:
            st.info("This dataset has no categorical columns.")
        else:
            top_n = st.slider("Top categories to show", min_value=3, max_value=30, value=10, key=f"all_cat_top_{dataset_key}")
            for feature in categorical[:6]:
                with st.expander(f"{feature} — category distribution", expanded=(categorical.index(feature) == 0)):
                    counts = frame[feature].fillna("Missing").astype(str).value_counts().head(top_n).rename_axis("Category").reset_index(name="Count")
                    fig = px.bar(counts, x="Category", y="Count", text="Count", title=f"Top {top_n} categories in {feature}")
                    fig.update_layout(xaxis_tickangle=-25)
                    plotly_chart(fig, f"all_cat_dist_{dataset_key}_{feature}")
                    st.dataframe(counts, hide_index=True, use_container_width=True)
            if len(categorical) > 6:
                st.caption("The first six categorical features are expanded to avoid rendering an excessive number of charts. Select any feature individually to inspect it.")

    with reltab:
        if len(numeric) >= 2:
            corr = frame[numeric].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Numeric feature correlation heatmap")
            plotly_chart(fig, f"all_corr_{dataset_key}")
            pairs = []
            for i, col_a in enumerate(numeric):
                for col_b in numeric[i + 1:]:
                    val = corr.loc[col_a, col_b]
                    if pd.notna(val):
                        pairs.append({"Feature 1": col_a, "Feature 2": col_b, "Correlation": float(val), "Absolute correlation": abs(float(val))})
            if pairs:
                pairs_df = pd.DataFrame(pairs).sort_values("Absolute correlation", ascending=False)
                st.markdown("**Strongest numeric correlations**")
                st.dataframe(pairs_df.head(10).drop(columns="Absolute correlation").round(3), hide_index=True, use_container_width=True)
        else:
            st.info("At least two numeric columns are required for correlation analysis.")
        if numeric and categorical:
            st.markdown("**Compare a numeric feature by category**")
            c1, c2 = st.columns(2)
            with c1:
                num_pick = st.selectbox("Numeric feature", numeric, key=f"all_group_num_{dataset_key}")
            with c2:
                cat_pick = st.selectbox("Grouping feature", categorical, key=f"all_group_cat_{dataset_key}")
            group = frame[[num_pick, cat_pick]].copy()
            group[cat_pick] = group[cat_pick].fillna("Missing").astype(str)
            top_cats = group[cat_pick].value_counts().head(12).index
            group = group[group[cat_pick].isin(top_cats)]
            fig = px.box(group, x=cat_pick, y=num_pick, points=False, title=f"{num_pick} by {cat_pick} (top 12 groups)")
            fig.update_layout(xaxis_tickangle=-25)
            plotly_chart(fig, f"all_group_box_{dataset_key}")

    with summtab:
        if numeric:
            st.markdown("**Numeric summary statistics**")
            st.dataframe(frame[numeric].describe().T, use_container_width=True)
            outlier_rows = []
            for col in numeric:
                s = frame[col].dropna()
                if len(s) == 0:
                    continue
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_rows.append({"Feature": col, "Lower bound": low, "Upper bound": high, "Potential outliers": int(((s < low) | (s > high)).sum())})
            st.markdown("**IQR-based potential outliers**")
            st.dataframe(pd.DataFrame(outlier_rows).round(3), hide_index=True, use_container_width=True)
        if categorical:
            st.markdown("**Categorical summary**")
            cat_summary = pd.DataFrame({
                "Feature": categorical,
                "Unique categories": [frame[c].nunique(dropna=True) for c in categorical],
                "Missing values": [int(frame[c].isna().sum()) for c in categorical],
                "Most common value": [frame[c].mode(dropna=True).iloc[0] if not frame[c].mode(dropna=True).empty else "—" for c in categorical],
            })
            st.dataframe(cat_summary, hide_index=True, use_container_width=True)


def render_feature_explorer() -> None:
    header("Feature Explorer", "Inspect feature distributions, data quality, and relationships without modifying the source CSV.")
    dataset_key = display_dataset_key()
    frame = safe_load(dataset_key)
    if frame is None:
        return
    st.caption(f"Source: `{DATASET_PATHS[dataset_key].relative_to(Path(__file__).resolve().parent)}` • {len(frame):,} rows • {frame.shape[1]} columns")
    selected = st.selectbox("Feature", ["All Features"] + list(frame.columns), key=f"feature_pick_{dataset_key}")
    if selected == "All Features":
        render_all_features(frame, dataset_key)
        return

    series = frame[selected]
    is_numeric = pd.api.types.is_numeric_dtype(series)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data type", str(series.dtype))
    c2.metric("Missing", f"{int(series.isna().sum()):,}")
    c3.metric("Unique values", f"{series.nunique(dropna=True):,}")
    c4.metric("Missing (%)", f"{100 * series.isna().mean():.2f}%")
    if is_numeric:
        modes = ["Histogram", "Box plot", "Violin plot", "Density histogram", "Values by record order"]
        chart = st.selectbox("Visualization", modes, key=f"num_viz_{dataset_key}_{selected}")
        if chart == "Histogram":
            fig = px.histogram(frame, x=selected, nbins=40, marginal="box", title=f"Distribution of {selected}")
        elif chart == "Box plot":
            fig = px.box(frame, y=selected, points="outliers", title=f"Box plot — {selected}")
        elif chart == "Violin plot":
            fig = px.violin(frame, y=selected, box=True, points="outliers", title=f"Violin plot — {selected}")
        elif chart == "Density histogram":
            fig = px.histogram(frame, x=selected, nbins=40, histnorm="probability density", title=f"Density histogram — {selected}")
        else:
            plot_frame = frame[[selected]].copy()
            plot_frame["Record"] = np.arange(1, len(plot_frame) + 1)
            if len(plot_frame) > 3000:
                plot_frame = plot_frame.iloc[np.linspace(0, len(plot_frame) - 1, 3000, dtype=int)]
                st.caption(f"Displaying a sample of {len(plot_frame):,} records to keep the plot responsive; statistics use the full column.")
            fig = px.line(plot_frame, x="Record", y=selected, title=f"{selected} by record order")
        plotly_chart(fig, f"feature_single_{dataset_key}_{selected}_{chart}")
        st.subheader("Summary statistics")
        st.dataframe(series.describe().to_frame("Value"), use_container_width=True)
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        st.caption(f"IQR bounds: {fmt_num(low)} to {fmt_num(high)} • Potential outliers: {int(((series < low) | (series > high)).sum())}")
    else:
        st.markdown("**Categorical analysis**")
        top_n = st.slider("Top-N categories", 3, 30, 10, key=f"topn_{dataset_key}_{selected}")
        display_type = st.selectbox("Visualization", ["Frequency bar chart", "Donut chart", "Category proportions"], key=f"cat_viz_{dataset_key}_{selected}")
        values = series.fillna("Missing").astype(str)
        counts = values.value_counts()
        top = counts.head(top_n).copy()
        if len(counts) > top_n:
            top.loc["Other"] = counts.iloc[top_n:].sum()
        top_df = top.rename_axis("Category").reset_index(name="Count")
        if display_type == "Donut chart":
            fig = px.pie(top_df, values="Count", names="Category", hole=0.55, title=f"Category share — {selected}")
        elif display_type == "Category proportions":
            top_df["Proportion (%)"] = 100 * top_df["Count"] / max(1, len(values))
            fig = px.bar(top_df, x="Category", y="Proportion (%)", text="Proportion (%)", title=f"Category proportions — {selected}")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        else:
            fig = px.bar(top_df, x="Category", y="Count", text="Count", title=f"Category frequency — {selected}")
        fig.update_layout(xaxis_tickangle=-25)
        plotly_chart(fig, f"category_single_{dataset_key}_{selected}_{display_type}")
        st.markdown("**Frequency table**")
        st.dataframe(top_df, hide_index=True, use_container_width=True)
    st.markdown("**Missingness for this feature**")
    missing_df = pd.DataFrame({"Status": ["Present", "Missing"], "Count": [int(series.notna().sum()), int(series.isna().sum())]})
    fig = px.bar(missing_df, x="Status", y="Count", color="Status", text="Count", title=f"Missing-value distribution — {selected}")
    plotly_chart(fig, f"feature_missing_{dataset_key}_{selected}")


def render_metric_comparison(results: list[dict], metric_options: list[str], key: str, lower_is_better: set[str] | None = None) -> None:
    if not results:
        st.info("Run a model evaluation to view metrics.")
        return
    rows = [r["metrics"] for r in results]
    df = pd.DataFrame(rows)
    st.dataframe(df.round(4), hide_index=True, use_container_width=True)
    metric = st.selectbox("Metric to visualize", metric_options, key=f"metric_dropdown_{key}")
    chart_df = df[["Algorithm", metric]].dropna().copy()
    if chart_df.empty:
        st.warning(f"No valid values are available for {metric}.")
        return
    lower_is_better = lower_is_better or set()
    chart_df = chart_df.sort_values(metric, ascending=metric in lower_is_better)
    fig = px.bar(chart_df, x="Algorithm", y=metric, color="Algorithm", text=metric, title=f"Model comparison — {metric}")
    if metric in lower_is_better:
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    elif "time" in metric.lower():
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    else:
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(showlegend=False, xaxis_tickangle=-20, yaxis_title=metric)
    plotly_chart(fig, f"metric_bar_{key}_{metric}")


def render_classification_detail(result: dict, model_name: str, key: str) -> None:
    metrics = result["metrics"]
    st.subheader(f"Detailed evaluation — {model_name}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Held-out accuracy", f"{metrics['Accuracy']:.3f}")
    k2.metric("Weighted F1", f"{metrics['Weighted F1-score']:.3f}")
    k3.metric("ROC-AUC", fmt_num(metrics["ROC-AUC"]))
    k4.metric("Training time", f"{metrics['Training time (s)']:.3f}s")

    cm, other = st.columns([1, 1.2])
    with cm:
        fig = px.imshow(
            result["confusion_matrix"], text_auto=True, aspect="auto", color_continuous_scale="Blues",
            x=[str(v) for v in result["class_labels"]], y=[str(v) for v in result["class_labels"]],
            labels={"x": "Predicted label", "y": "Actual label", "color": "Count"}, title="Confusion matrix",
        )
        plotly_chart(fig, f"confusion_{key}_{model_name}")
    with other:
        st.markdown("**Classification report**")
        st.dataframe(result["report"].round(3), use_container_width=True)
        st.caption("The table is computed from held-out test predictions, not training predictions.")

    if result["fpr"] is not None and result["tpr"] is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result["fpr"], y=result["tpr"], mode="lines", name=f"ROC (AUC {metrics['ROC-AUC']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Chance baseline"))
        fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", yaxis=dict(range=[0, 1.02]), xaxis=dict(range=[0, 1.02]))
        plotly_chart(fig, f"roc_{key}_{model_name}")
    else:
        st.info("ROC curve is unavailable for this run because the model or target did not provide valid score values.")

    actual_pred = result["actual_pred"].copy()
    dist = pd.concat([
        actual_pred["Actual"].astype(str).value_counts().rename_axis("Class").reset_index(name="Count").assign(Source="Actual"),
        actual_pred["Predicted"].astype(str).value_counts().rename_axis("Class").reset_index(name="Count").assign(Source="Predicted"),
    ], ignore_index=True)
    fig = px.bar(dist, x="Class", y="Count", color="Source", barmode="group", title="Actual vs predicted class counts")
    plotly_chart(fig, f"class_distribution_{key}_{model_name}")
    if not result["feature_importance"].empty:
        with st.expander("Native feature importance / coefficient magnitude"):
            fi = result["feature_importance"].head(20).sort_values("Importance")
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Top model features")
            plotly_chart(fig, f"class_feature_importance_{key}_{model_name}")


def render_classification() -> None:
    header("Classification Lab", "Compare supervised classifiers with a reproducible, stratified held-out evaluation.")
    dataset_key = "Diabetes classification"
    frame = safe_load(dataset_key)
    if frame is None:
        return
    target = TARGETS[dataset_key]
    st.info(f"Dataset: `{DATASET_PATHS[dataset_key].name}` • Target: `{target}` • Fixed test split: 80:20 • random_state=42 • Preprocessing is fitted inside each training pipeline.")
    raw_counts = frame[target].value_counts().sort_index().rename_axis("Class").reset_index(name="Count")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{len(frame):,}")
    c2.metric("Input features", frame.shape[1] - 1)
    c3.metric("Positive class rate", f"{100 * (frame[target] == 1).mean():.1f}%" if set(frame[target].dropna().unique()).issubset({0, 1}) else "See distribution")
    c4.metric("Exact duplicates", f"{frame.duplicated().sum():,}")
    fig = px.bar(raw_counts, x="Class", y="Count", text="Count", title="Original target class distribution")
    plotly_chart(fig, "classification_target_distribution")
    st.caption("Severe class imbalance can make accuracy look strong while minority-class recall is weak. Inspect macro/weighted scores and the confusion matrix together.")

    all_features = [c for c in frame.columns if c != target]
    with st.expander("Feature controls / leakage review", expanded=False):
        st.write("The target column is excluded automatically. Exclude any predictor you believe may reveal the target directly and compare the held-out scores.")
        excluded = st.multiselect("Exclude input features", all_features, default=[], key="class_excluded")
        st.caption("Numeric features strongly correlated with a target are not removed automatically; high correlation can be informative, but should be reviewed for direct or indirect leakage.")
    excluded_tuple = tuple(sorted(excluded))
    mode = st.radio("Evaluation mode", ["Single algorithm", "Compare multiple algorithms"], horizontal=True, key="classification_mode")
    if mode == "Single algorithm":
        selected = st.selectbox("Algorithm", CLASSIFIERS, key="classification_single_algorithm")
        algorithms = [selected]
    else:
        algorithms = st.multiselect("Algorithms to compare", CLASSIFIERS, default=CLASSIFIERS[:4], key="classification_algorithms")
    run = st.button("Run classification evaluation", type="primary", key="run_classification")

    if run:
        if not algorithms:
            st.warning("Select at least one algorithm.")
        else:
            fresh = []
            progress = st.progress(0, text="Preparing evaluation…")
            errors = []
            for index, name in enumerate(algorithms):
                try:
                    with st.spinner(f"Training {name} on the training split…"):
                        res = cached_classification(dataset_key, name, excluded_tuple, _dataset_stamp(dataset_key))
                    fresh.append({"name": name, "result": res})
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
                progress.progress((index + 1) / len(algorithms), text=f"Evaluated {index + 1} of {len(algorithms)} selected model(s)")
            progress.empty()
            st.session_state["classification_run"] = {"items": fresh, "excluded": excluded_tuple, "algorithms": tuple(algorithms)}
            st.session_state["classification_errors"] = errors
    saved = st.session_state.get("classification_run")
    if saved:
        if saved["excluded"] != excluded_tuple or tuple(algorithms) != saved["algorithms"]:
            st.warning("The controls differ from the last completed run. Click Run classification evaluation to refresh the displayed results.")
        errors = st.session_state.get("classification_errors", [])
        for error in errors:
            st.error(error)
        if saved["items"]:
            st.subheader("Held-out model comparison")
            results = [item["result"] for item in saved["items"]]
            metric_options = ["Accuracy", "Weighted precision", "Macro precision", "Weighted recall", "Macro recall", "Weighted F1-score", "Macro F1-score", "ROC-AUC", "Training time (s)", "Prediction time (s)"]
            render_metric_comparison(results, metric_options, "classification", lower_is_better={"Training time (s)", "Prediction time (s)"})
            chosen = st.selectbox("Inspect model details", [item["name"] for item in saved["items"]], key="class_detail_model")
            result = next(item["result"] for item in saved["items"] if item["name"] == chosen)
            render_classification_detail(result, chosen, "classification")
            metrics_df = pd.DataFrame([r["metrics"] for r in results])
            st.download_button("Download comparison CSV", data=metrics_df.to_csv(index=False).encode("utf-8"), file_name="classification_comparison.csv", mime="text/csv")
            if any(abs(r["metrics"]["Accuracy"] - 1) < 1e-12 for r in results):
                st.warning("At least one model has perfect held-out accuracy. This is displayed as calculated, not altered; investigate data leakage, duplicates, split logic, and label quality before interpreting it as reliable generalization.")


def render_regression_detail(result: dict, model_name: str) -> None:
    metrics = result["metrics"]
    st.subheader(f"Detailed evaluation — {model_name}")
    a, b, c, d = st.columns(4)
    a.metric("Held-out R²", f"{metrics['R²']:.3f}")
    b.metric("RMSE", f"{metrics['RMSE']:,.2f}")
    c.metric("MAE", f"{metrics['MAE']:,.2f}")
    d.metric("Training time", f"{metrics['Training time (s)']:.3f}s")

    pred = result["actual_pred"]
    left, right = st.columns(2)
    with left:
        fig = px.scatter(pred, x="Actual", y="Predicted", opacity=0.7, title="Actual vs predicted values")
        low = min(pred["Actual"].min(), pred["Predicted"].min())
        high = max(pred["Actual"].max(), pred["Predicted"].max())
        fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", line=dict(dash="dash"), name="Ideal prediction"))
        plotly_chart(fig, f"reg_actual_pred_{model_name}")
    with right:
        fig = px.histogram(pred, x="Residual", nbins=35, marginal="box", title="Residual distribution")
        plotly_chart(fig, f"reg_residual_hist_{model_name}")
    fig = px.scatter(pred, x="Predicted", y="Residual", opacity=0.7, title="Residuals vs predicted values")
    fig.add_hline(y=0, line_dash="dash")
    plotly_chart(fig, f"reg_residual_scatter_{model_name}")
    with st.expander("Sample of held-out actual and predicted values"):
        st.dataframe(pred.head(250).round(3), hide_index=True, use_container_width=True)
        st.caption(f"Showing at most 250 of {len(pred):,} test predictions; all test rows are used for metric calculation.")
    if not result["feature_importance"].empty:
        with st.expander("Native feature importance / coefficient magnitude"):
            fi = result["feature_importance"].head(20).sort_values("Importance")
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Top model features")
            plotly_chart(fig, f"reg_feature_importance_{model_name}")
    else:
        st.caption("This model does not expose a native feature-importance or coefficient measure.")


def render_regression() -> None:
    header("Regression Lab", "Compare models for insurance charge prediction and inspect errors, residuals, and generalization.")
    dataset_key = "Insurance regression"
    frame = safe_load(dataset_key)
    if frame is None:
        return
    target = TARGETS[dataset_key]
    valid_rows = int(frame[target].notna().sum())
    missing_target = int(frame[target].isna().sum())
    st.info(f"Dataset: `{DATASET_PATHS[dataset_key].name}` • Target: `{target}` • Fixed 80:20 split • random_state=42. Rows with a missing target ({missing_target}) are excluded from supervised evaluation; missing predictor values are imputed within the training pipeline.")
    a, b, c, d = st.columns(4)
    a.metric("Original records", f"{len(frame):,}")
    b.metric("Usable target rows", f"{valid_rows:,}")
    c.metric("Missing target rows", f"{missing_target:,}")
    d.metric("Features", frame.shape[1] - 1)
    with st.expander("Feature controls / exclude predictors", expanded=False):
        excluded = st.multiselect("Exclude input features", [c for c in frame.columns if c != target], default=[], key="reg_excluded")
    excluded_tuple = tuple(sorted(excluded))
    mode = st.radio("Evaluation mode", ["Single algorithm", "Compare multiple algorithms"], horizontal=True, key="regression_mode")
    if mode == "Single algorithm":
        algorithms = [st.selectbox("Algorithm", REGRESSORS, key="regression_single_algorithm")]
    else:
        algorithms = st.multiselect("Algorithms to compare", REGRESSORS, default=REGRESSORS[:4], key="regression_algorithms")
    if st.button("Run regression evaluation", type="primary", key="run_regression"):
        if not algorithms:
            st.warning("Select at least one algorithm.")
        else:
            fresh, errors = [], []
            progress = st.progress(0, text="Preparing regression evaluation…")
            for index, name in enumerate(algorithms):
                try:
                    with st.spinner(f"Training {name}…"):
                        res = cached_regression(dataset_key, name, excluded_tuple, _dataset_stamp(dataset_key))
                    fresh.append({"name": name, "result": res})
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
                progress.progress((index + 1) / len(algorithms), text=f"Evaluated {index + 1} of {len(algorithms)} selected model(s)")
            progress.empty()
            st.session_state["regression_run"] = {"items": fresh, "excluded": excluded_tuple, "algorithms": tuple(algorithms)}
            st.session_state["regression_errors"] = errors
    saved = st.session_state.get("regression_run")
    if saved:
        if saved["excluded"] != excluded_tuple or tuple(algorithms) != saved["algorithms"]:
            st.warning("The controls differ from the last completed run. Click Run regression evaluation to refresh the displayed results.")
        for error in st.session_state.get("regression_errors", []):
            st.error(error)
        if saved["items"]:
            results = [item["result"] for item in saved["items"]]
            st.subheader("Held-out regression comparison")
            render_metric_comparison(results, ["R²", "RMSE", "MAE", "Training time (s)", "Prediction time (s)"], "regression", lower_is_better={"RMSE", "MAE", "Training time (s)", "Prediction time (s)"})
            chosen = st.selectbox("Inspect model details", [item["name"] for item in saved["items"]], key="reg_detail_model")
            result = next(item["result"] for item in saved["items"] if item["name"] == chosen)
            render_regression_detail(result, chosen)
            metrics_df = pd.DataFrame([r["metrics"] for r in results])
            st.download_button("Download comparison CSV", data=metrics_df.to_csv(index=False).encode("utf-8"), file_name="regression_comparison.csv", mime="text/csv")


def render_clustering() -> None:
    header("Clustering Studio", "Explore patient groups using K-Means and Agglomerative clustering, with quantitative validation.")
    dataset_key = "Patient clustering"
    frame = safe_load(dataset_key)
    if frame is None:
        return
    a, b, c, d = st.columns(4)
    a.metric("Patient records", f"{len(frame):,}")
    b.metric("Numeric features", len(frame.select_dtypes(include=np.number).columns))
    c.metric("Missing cells", f"{frame.isna().sum().sum():,}")
    d.metric("Random seed", "42")
    st.caption("Clustering has no ground-truth target column. Scores measure internal cluster structure, not clinical validity.")
    clusters = st.slider("Number of clusters", min_value=2, max_value=8, value=3, key="cluster_count")
    include_tsne = st.checkbox("Calculate t-SNE visualization (adds computation)", value=False, key="cluster_tsne")
    if st.button("Run clustering comparison", type="primary", key="run_clustering"):
        with st.spinner("Imputing and scaling features, fitting clustering models, and calculating metrics…"):
            try:
                result = cached_clustering(clusters, include_tsne, _dataset_stamp(dataset_key))
                st.session_state["clustering_run"] = {"result": result, "clusters": clusters, "tsne": include_tsne}
            except Exception as exc:
                st.error(f"Clustering failed: {exc}")
    saved = st.session_state.get("clustering_run")
    if not saved:
        with st.expander("What the metrics mean"):
            st.markdown("- **Silhouette:** higher is generally better (range −1 to 1).\n- **Davies–Bouldin:** lower is better.\n- **Calinski–Harabasz:** higher is generally better.\nCompare all three and inspect visualizations; they can disagree.")
        return
    if saved["clusters"] != clusters or saved["tsne"] != include_tsne:
        st.warning("Your controls differ from the most recent result. Click Run clustering comparison to recalculate.")
    result = saved["result"]
    st.subheader("Algorithm comparison")
    st.dataframe(result["metrics"].round(4), hide_index=True, use_container_width=True)
    m1, m2, m3 = st.columns(3)
    metric_df = result["metrics"]
    m1.metric("Best Silhouette", str(metric_df.loc[metric_df["Silhouette Score"].idxmax(), "Algorithm"]))
    m2.metric("Lowest Davies–Bouldin", str(metric_df.loc[metric_df["Davies-Bouldin Index"].idxmin(), "Algorithm"]))
    m3.metric("Best Calinski–Harabasz", str(metric_df.loc[metric_df["Calinski-Harabasz Score"].idxmax(), "Algorithm"]))
    tab1, tab2, tab3, tab4 = st.tabs(["PCA plots", "Elbow curve", "Dendrogram", "Assignments & t-SNE"])
    with tab1:
        v = result["visualization"]
        for algorithm, color_col in [("K-Means", "K-Means"), ("Agglomerative (Ward)", "Agglomerative (Ward)")]:
            fig = px.scatter(v, x="PC1", y="PC2", color=color_col, opacity=0.78, title=f"PCA projection — {algorithm}", labels={"PC1": f"PC1 ({result['pca_variance'][0]*100:.1f}% variance)", "PC2": f"PC2 ({result['pca_variance'][1]*100:.1f}% variance)"})
            plotly_chart(fig, f"pca_{algorithm}_{clusters}")
        st.caption("PCA is a 2D projection used for visualization; clustering metrics are computed in the scaled original feature space.")
    with tab2:
        elbow = pd.DataFrame({"k": result["inertia_k"], "Inertia (WCSS)": result["inertia"]})
        fig = px.line(elbow, x="k", y="Inertia (WCSS)", markers=True, title="K-Means elbow curve")
        plotly_chart(fig, f"elbow_{clusters}")
        st.caption("Look for an elbow where additional clusters provide diminishing reductions in inertia; it is a heuristic, not a definitive rule.")
    with tab3:
        sample_n = min(250, len(result["scaled"]))
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(len(result["scaled"]), size=sample_n, replace=False))
        Z = linkage(result["scaled"][sample_idx], method="ward")
        dendro = dendrogram(Z, no_plot=True, no_labels=True)
        fig = go.Figure()
        for xs, ys in zip(dendro["icoord"], dendro["dcoord"]):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1), hoverinfo="skip", showlegend=False))
        fig.update_layout(title=f"Ward dendrogram (sample of {sample_n:,} rows)", xaxis_title="Sample observations", yaxis_title="Ward distance")
        plotly_chart(fig, f"dendrogram_{clusters}")
        st.caption("Dendrogram uses a deterministic sample for readability; the model evaluation above uses all rows.")
    with tab4:
        assignments = result["assignments"]
        st.dataframe(assignments.head(250), hide_index=True, use_container_width=True)
        st.download_button("Download full cluster assignments", assignments.to_csv(index=False).encode("utf-8"), file_name="clustering_assignments.csv", mime="text/csv")
        st.caption(f"Showing the first 250 of {len(assignments):,} rows here; the downloaded CSV includes all assignments.")
        if result["tsne"] is not None:
            ts = pd.DataFrame(result["tsne"], columns=["t-SNE 1", "t-SNE 2"])
            ts["K-Means cluster"] = result["labels"]["K-Means"].astype(str)
            fig = px.scatter(ts, x="t-SNE 1", y="t-SNE 2", color="K-Means cluster", title="t-SNE projection — colored by K-Means labels")
            plotly_chart(fig, f"tsne_{clusters}")
        else:
            st.info("Enable the t-SNE checkbox and rerun if you want the additional 2D embedding.")


def _input_control(form, frame: pd.DataFrame, column: str):
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series):
        clean = series.dropna()
        default = float(clean.median()) if not clean.empty else 0.0
        is_integer = pd.api.types.is_integer_dtype(series)
        min_val = float(clean.min()) if not clean.empty else None
        max_val = float(clean.max()) if not clean.empty else None
        if is_integer:
            value = form.number_input(column, value=int(round(default)), step=1, help=f"Training range: {min_val:g} to {max_val:g}" if min_val is not None else None)
            return int(value)
        step = max(0.01, round(abs(default) / 100, 2))
        value = form.number_input(column, value=default, step=float(step), format="%.3f", help=f"Training range: {min_val:g} to {max_val:g}" if min_val is not None else None)
        return float(value)
    options = sorted(series.dropna().astype(str).unique().tolist())
    if not options:
        options = ["Unknown"]
    return form.selectbox(column, options, help="Choose a category present in the training dataset.")


def render_predict_status() -> None:
    header("Predict Status", "Train with the project's fixed datasets, then enter a row of feature values to request one prediction.")
    task = st.radio("Prediction task", ["Diabetes classification", "Insurance regression"], horizontal=True, key="prediction_task")
    frame = safe_load(task)
    if frame is None:
        return
    target = TARGETS[task]
    if task == "Diabetes classification":
        options = CLASSIFIERS
        selected = st.selectbox("Classifier", options, key="predict_class_model")
        st.caption("The model is evaluated with a held-out stratified split. The entered row is an individual demonstration prediction, not medical advice.")
    else:
        selected = st.selectbox("Regressor", REGRESSORS, key="predict_reg_model")
        st.caption("The model is evaluated on held-out insurance records. Predictions are estimates for demonstration and should not be treated as financial advice.")

    st.subheader("Enter predictor values")
    input_columns = [c for c in frame.columns if c != target]
    with st.form("prediction_form"):
        vals = {}
        cols = st.columns(2)
        for i, column in enumerate(input_columns):
            with cols[i % 2]:
                vals[column] = _input_control(st, frame, column)
        submitted = st.form_submit_button("Train / load model and predict", type="primary", use_container_width=True)
    if submitted:
        try:
            with st.spinner("Preparing the evaluation model and calculating the prediction…"):
                if task == "Diabetes classification":
                    result = cached_classification(task, selected, tuple(), _dataset_stamp(task))
                else:
                    result = cached_regression(task, selected, tuple(), _dataset_stamp(task))
                new_row = pd.DataFrame([{c: vals[c] for c in result["input_columns"]}])
                prediction = result["model"].predict(new_row)[0]
            st.subheader("Prediction result")
            if task == "Diabetes classification":
                label_map = {0: "No diabetes predicted", 1: "Diabetes predicted"}
                pred_label = label_map.get(prediction, str(prediction))
                (st.warning if prediction == 1 else st.success)(pred_label)
                st.metric("Predicted class", str(prediction))
                if hasattr(result["model"], "predict_proba"):
                    probabilities = result["model"].predict_proba(new_row)[0]
                    classes = result["model"].named_steps["model"].classes_
                    prob_df = pd.DataFrame({"Class": [str(x) for x in classes], "Probability": probabilities})
                    fig = px.bar(prob_df, x="Class", y="Probability", text="Probability", range_y=[0, 1], title="Predicted class probabilities")
                    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
                    plotly_chart(fig, "single_prediction_probability")
                st.caption("This is an educational ML project prediction and is not a clinical diagnosis.")
                st.write({"Held-out test accuracy": round(result["metrics"]["Accuracy"], 4), "Held-out weighted F1": round(result["metrics"]["Weighted F1-score"], 4), "ROC-AUC": fmt_num(result["metrics"]["ROC-AUC"])})
            else:
                st.success(f"Predicted insurance charges: {float(prediction):,.2f}")
                st.write({"Held-out test R²": round(result["metrics"]["R²"], 4), "Held-out RMSE": round(result["metrics"]["RMSE"], 2), "Held-out MAE": round(result["metrics"]["MAE"], 2)})
        except Exception as exc:
            st.error(f"Could not generate the prediction: {exc}")


def render_about() -> None:
    header("About Bias Breakers", "Project overview, data sources, algorithms, evaluation rules, and limitations.")
    st.markdown("""
    **Bias Breakers** is a machine-learning capstone project with three complementary analysis tracks.

    - **Classification:** diabetes label prediction using the included `diabetes_prediction_dataset_2000.csv` dataset.
    - **Regression:** insurance charges estimation using `regression_insurance.csv`.
    - **Clustering:** unsupervised patient grouping using `clustering_patients.csv`.

    ### Algorithms available
    - **Classification:** Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, Decision Tree, Random Forest, Support Vector Machine, AdaBoost, Gradient Boosting, Bagging, and MLP Classifier.
    - **Regression:** Linear, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, Gradient Boosting, Support Vector Regression, and KNN Regression.
    - **Clustering:** K-Means and Agglomerative (Ward) clustering.

    ### Evaluation approach
    - Supervised models use a reproducible 80:20 held-out split with `random_state=42`; classification uses stratification when possible.
    - Imputation, categorical encoding, and numeric scaling are fitted within the training pipeline to reduce leakage.
    - Classification reports accuracy, weighted and macro precision/recall/F1, ROC-AUC when valid, plus a confusion matrix and classification report.
    - Regression reports R², RMSE, MAE, train/prediction time, actual-vs-predicted values, and residual diagnostics.
    - Clustering reports Silhouette, Davies–Bouldin, and Calinski–Harabasz metrics. These internal metrics do not prove that clusters are clinically meaningful.

    ### Limitations
    - The datasets are used as provided in this repository. The dashboard does not claim external validation or production readiness.
    - A high score is not guaranteed. Perfect metrics, if encountered, are displayed as calculated and should trigger a data-leakage and data-quality review.
    - The diabetes prediction is for educational demonstration only, not medical advice or diagnosis.
    - This uploaded project contains diabetes, insurance, and patient-clustering CSV files; it does not include the EV charging Excel dataset mentioned in a separate brief.
    """)


def main() -> None:
    with st.sidebar:
        st.markdown("# 🧭 Bias Breakers")
        st.caption("ML Analytics Workspace")
        st.markdown("---")
        page = st.radio(
            "Navigate",
            ["Overview", "Feature Explorer", "Predict Status", "Classification", "Regression", "Clustering", "About Project"],
            label_visibility="collapsed",
            key="main_navigation",
        )
        st.markdown("---")
        st.markdown("**Fixed project datasets**")
        st.caption("Diabetes • Insurance • Patient clustering")
        st.caption("Seed: 42")

    if page == "Overview":
        render_overview()
    elif page == "Feature Explorer":
        render_feature_explorer()
    elif page == "Predict Status":
        render_predict_status()
    elif page == "Classification":
        render_classification()
    elif page == "Regression":
        render_regression()
    elif page == "Clustering":
        render_clustering()
    else:
        render_about()

    st.markdown("---")
    st.caption("Bias Breakers · Machine Learning Capstone · Models train only when requested; fixed CSV datasets are cached between reruns.")


if __name__ == "__main__":
    main()
