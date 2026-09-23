"""
Customer Churn Analysis & Retention Dashboard
=============================================
Single-file Streamlit app for Telco Customer Churn analysis.
Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv
Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Pages:
  1. Executive Overview  – KPIs, churn split, contract breakdown, key insights
  2. Churn Drivers       – Feature-level churn charts with data-driven explanations
  3. Risk & Actions      – ML models, top-20 at-risk table, single customer form,
                           Risks / Opportunities / Recommended Actions

Run: streamlit run churn_retention_app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Dashboard",
    page_icon="📡",
    layout="wide",
)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
CSV_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
KAGGLE_LINK  = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"

# colour palette (consistent across all charts)
CHURN_COLORS   = {"Churned": "#EF4444", "Retained": "#22C55E"}
ACCENT         = "#3B82F6"
SECONDARY      = "#7C3AED"

# ──────────────────────────────────────────────
# DATA LOADING & CLEANING
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & cleaning data …")
def load_data() -> pd.DataFrame:
    """Load the CSV, clean it, and return a ready-to-use DataFrame."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, CSV_FILENAME)

    if not os.path.exists(csv_path):
        st.error(
            f"Dataset not found at `{csv_path}`.\n\n"
            f"Download it from Kaggle: {KAGGLE_LINK}"
        )
        st.stop()

    df = pd.read_csv(csv_path)

    # --- TotalCharges: blank values exist for new customers (tenure == 0)
    #     We fill them with 0.0 (their total spend IS zero; dropping 11 rows
    #     would silently remove valid new-customer records).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # --- Map Churn to binary integer
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # --- Tenure groups for binned charts
    bins   = [-1, 12, 24, 48, 60, 72]
    labels = ["0–12 mo", "13–24 mo", "25–48 mo", "49–60 mo", "61–72 mo"]
    df["TenureGroup"] = pd.cut(df["tenure"], bins=bins, labels=labels)

    return df


# ──────────────────────────────────────────────
# MODEL TRAINING
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models …")
def train_models(df: pd.DataFrame):
    """
    Encode features, train Logistic Regression & Random Forest,
    return both models, scaler, encoders, feature names, test sets,
    and evaluation metrics.
    """
    model_df = df.drop(columns=["customerID", "TenureGroup"], errors="ignore").copy()

    # Encode binary Yes/No and categorical columns
    label_encoders = {}
    for col in model_df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        model_df[col] = le.fit_transform(model_df[col].astype(str))
        label_encoders[col] = le

    feature_cols = [c for c in model_df.columns if c != "Churn"]
    X = model_df[feature_cols].values
    y = model_df["Churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Scale for Logistic Regression
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # --- Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_sc, y_train)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]
    lr_pred  = (lr_proba >= 0.5).astype(int)

    lr_metrics = {
        "Accuracy":  round(accuracy_score(y_test, lr_pred), 4),
        "Precision": round(precision_score(y_test, lr_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, lr_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, lr_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, lr_proba), 4),
    }

    # --- Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred  = (rf_proba >= 0.5).astype(int)

    rf_metrics = {
        "Accuracy":  round(accuracy_score(y_test, rf_pred), 4),
        "Precision": round(precision_score(y_test, rf_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, rf_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, rf_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, rf_proba), 4),
    }

    # Determine best model by ROC-AUC
    if rf_metrics["ROC-AUC"] >= lr_metrics["ROC-AUC"]:
        best_name    = "Random Forest"
        best_model   = rf
        best_proba   = rf_proba
        best_pred    = rf_pred
        best_metrics = rf_metrics
        best_scaled  = False
    else:
        best_name    = "Logistic Regression"
        best_model   = lr
        best_proba   = lr_proba
        best_pred    = lr_pred
        best_metrics = lr_metrics
        best_scaled  = True

    # Confusion matrix for best model
    cm = confusion_matrix(y_test, best_pred)

    # Feature importances
    if best_scaled:
        importances = np.abs(best_model.coef_[0])
    else:
        importances = best_model.feature_importances_

    feat_imp = (
        pd.DataFrame({"Feature": feature_cols, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    # Full dataset churn probabilities for top-20 at-risk table
    X_all = model_df[feature_cols].values
    if best_scaled:
        X_all_sc = scaler.transform(X_all)
        all_proba = best_model.predict_proba(X_all_sc)[:, 1]
    else:
        all_proba = best_model.predict_proba(X_all)[:, 1]

    return {
        "lr_metrics":    lr_metrics,
        "rf_metrics":    rf_metrics,
        "best_name":     best_name,
        "best_metrics":  best_metrics,
        "cm":            cm,
        "feat_imp":      feat_imp,
        "all_proba":     all_proba,
        "feature_cols":  feature_cols,
        "label_encoders": label_encoders,
        "scaler":        scaler,
        "best_model":    best_model,
        "best_scaled":   best_scaled,
        "X_test":        X_test,
        "y_test":        y_test,
    }


# ──────────────────────────────────────────────
# HELPER: KPI METRIC CARD
# ──────────────────────────────────────────────
def kpi_card(col, label: str, value: str, delta: str = "", color: str = "#1F2937"):
    col.markdown(
        f"""
        <div style="background:#F9FAFB;border:1px solid #E5E7EB;border-radius:10px;
                    padding:18px 20px;text-align:center;">
            <div style="font-size:13px;color:#6B7280;margin-bottom:4px;">{label}</div>
            <div style="font-size:26px;font-weight:700;color:{color};">{value}</div>
            <div style="font-size:12px;color:#9CA3AF;margin-top:4px;">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# PAGE 1 – EXECUTIVE OVERVIEW
# ──────────────────────────────────────────────
def page_executive_overview(df: pd.DataFrame, results: dict):
    st.title("📊 Executive Overview")
    st.markdown("High-level snapshot of churn health and key business signals.")

    total      = len(df)
    churned    = int(df["Churn"].sum())
    retained   = total - churned
    churn_rate = churned / total * 100
    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure  = df["tenure"].mean()
    revenue_at_risk = df[df["Churn"] == 1]["MonthlyCharges"].sum()
    roc_auc     = results["best_metrics"]["ROC-AUC"]

    # ── KPI Cards ──────────────────────────────
    st.subheader("Key Performance Indicators")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_card(c1, "Total Customers",       f"{total:,}")
    kpi_card(c2, "Churn Rate",            f"{churn_rate:.1f}%",     color="#EF4444")
    kpi_card(c3, "Avg Monthly Charges",   f"${avg_monthly:.2f}")
    kpi_card(c4, "Avg Tenure",            f"{avg_tenure:.1f} mo")
    kpi_card(c5, "Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}", color="#F59E0B")
    kpi_card(c6, "Model ROC-AUC",         f"{roc_auc:.4f}",         color="#3B82F6")

    st.markdown("---")

    # ── Churn vs Retained Donut ────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Churn vs Retained")
        donut = go.Figure(go.Pie(
            labels=["Churned", "Retained"],
            values=[churned, retained],
            hole=0.55,
            marker_colors=[CHURN_COLORS["Churned"], CHURN_COLORS["Retained"]],
            textinfo="label+percent",
        ))
        donut.update_layout(
            showlegend=True, height=340,
            annotations=[dict(text=f"{churn_rate:.1f}%<br>Churn", x=0.5, y=0.5,
                              font_size=16, showarrow=False)],
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(donut, width="stretch")

    # ── Churn Rate by Contract ─────────────────
    with right:
        st.subheader("Churn Rate by Contract Type")
        contract_stats = (
            df.groupby("Contract")["Churn"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "ChurnRate", "count": "Count"})
        )
        contract_stats["ChurnPct"] = (contract_stats["ChurnRate"] * 100).round(1)
        bar = px.bar(
            contract_stats, x="Contract", y="ChurnPct",
            text=contract_stats["ChurnPct"].apply(lambda v: f"{v:.1f}%"),
            color="ChurnPct",
            color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
            labels={"ChurnPct": "Churn Rate (%)", "Contract": "Contract Type"},
        )
        bar.update_traces(textposition="outside")
        bar.update_layout(height=340, showlegend=False, margin=dict(t=10, b=10),
                          coloraxis_showscale=False)
        st.plotly_chart(bar, width="stretch")

    st.markdown("---")

    # ── Key Insights ──────────────────────────
    st.subheader("🔍 Key Insights")
    mtm_rate = (
        df[df["Contract"] == "Month-to-month"]["Churn"].mean() * 100
    )
    fiber_rate = (
        df[df["InternetService"] == "Fiber optic"]["Churn"].mean() * 100
    )
    senior_rate  = df[df["SeniorCitizen"] == 1]["Churn"].mean() * 100
    no_tech_rate = df[df["TechSupport"] == "No"]["Churn"].mean() * 100

    i1, i2, i3, i4 = st.columns(4)
    i1.info(
        f"**Month-to-Month Churn: {mtm_rate:.1f}%**\n\n"
        "Month-to-month subscribers churn at nearly 3× the rate of annual contract customers."
    )
    i2.warning(
        f"**Fiber Optic Churn: {fiber_rate:.1f}%**\n\n"
        "Fiber optic customers churn at the highest rate among internet service types, "
        "suggesting pricing or service-quality concerns."
    )
    i3.error(
        f"**Senior Citizens: {senior_rate:.1f}%**\n\n"
        "Senior customers have a significantly higher churn rate, likely due to "
        "price sensitivity and less tech familiarity."
    )
    i4.success(
        f"**No Tech Support: {no_tech_rate:.1f}%**\n\n"
        "Customers without tech support churn at {no_tech_rate:.1f}%, nearly double "
        "those with support — a clear retention lever."
    )


# ──────────────────────────────────────────────
# PAGE 2 – CHURN DRIVERS
# ──────────────────────────────────────────────
def page_churn_drivers(df: pd.DataFrame):
    st.title("🔎 Churn Drivers")
    st.markdown(
        "Deep-dive into which features drive churn. Every percentage is computed "
        "directly from the dataset."
    )

    def churn_bar(group_col: str, title: str):
        stats = (
            df.groupby(group_col)["Churn"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "ChurnRate", "count": "N"})
        )
        stats["ChurnPct"] = (stats["ChurnRate"] * 100).round(1)
        stats = stats.sort_values("ChurnPct", ascending=False)
        fig = px.bar(
            stats, x=group_col, y="ChurnPct",
            text=stats["ChurnPct"].apply(lambda v: f"{v:.1f}%"),
            color="ChurnPct",
            color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
            labels={"ChurnPct": "Churn Rate (%)"},
            title=title,
            custom_data=[stats["N"].values],
        )
        fig.update_traces(
            textposition="outside",
            hovertemplate="%{x}<br>Churn Rate: %{y:.1f}%<br>N: %{customdata[0]}<extra></extra>",
        )
        fig.update_layout(
            height=350, showlegend=False, coloraxis_showscale=False,
            margin=dict(t=40, b=10),
        )
        return fig, stats

    # ── Row 1: Internet Service + Contract ────
    col1, col2 = st.columns(2)
    with col1:
        fig, stats = churn_bar("InternetService", "Churn Rate by Internet Service")
        st.plotly_chart(fig, width="stretch")
        top = stats.iloc[0]
        bot = stats.iloc[-1]
        st.caption(
            f"**Why:** {top[stats.columns[0]]} customers churn at **{top['ChurnPct']}%**, "
            f"vs {bot['ChurnPct']}% for {bot[stats.columns[0]]} — a gap of "
            f"{top['ChurnPct'] - bot['ChurnPct']:.1f} pp. Fiber optic's premium price "
            "combined with no automatic contract lock-in appears to drive dissatisfaction."
        )

    with col2:
        fig, stats = churn_bar("Contract", "Churn Rate by Contract Type")
        st.plotly_chart(fig, width="stretch")
        mtm = stats[stats["Contract"] == "Month-to-month"]["ChurnPct"].values[0]
        two_yr = stats[stats["Contract"] == "Two year"]["ChurnPct"].values[0]
        st.caption(
            f"**Why:** Month-to-month customers churn at **{mtm:.1f}%** vs only "
            f"{two_yr:.1f}% on two-year contracts. Zero switching cost and flexible "
            "billing make it easy for dissatisfied customers to leave immediately."
        )

    # ── Row 2: Payment Method + Tenure Group ──
    col3, col4 = st.columns(2)
    with col3:
        fig, stats = churn_bar("PaymentMethod", "Churn Rate by Payment Method")
        st.plotly_chart(fig, width="stretch")
        top = stats.iloc[0]
        st.caption(
            f"**Why:** {top['PaymentMethod']} has the highest churn at "
            f"**{top['ChurnPct']}%**. Electronic-check payers tend to be less "
            "committed customers who actively manage each bill — making them "
            "more responsive to a competitor's promotional offer."
        )

    with col4:
        tg = (
            df.groupby("TenureGroup", observed=True)["Churn"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "ChurnRate", "count": "N"})
        )
        tg["ChurnPct"] = (tg["ChurnRate"] * 100).round(1)
        fig_tg = px.bar(
            tg, x="TenureGroup", y="ChurnPct",
            text=tg["ChurnPct"].apply(lambda v: f"{v:.1f}%"),
            color="ChurnPct",
            color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
            labels={"ChurnPct": "Churn Rate (%)", "TenureGroup": "Tenure Group"},
            title="Churn Rate by Tenure Group",
        )
        fig_tg.update_traces(textposition="outside")
        fig_tg.update_layout(
            height=350, showlegend=False, coloraxis_showscale=False,
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig_tg, width="stretch")
        early = tg.iloc[0]
        late  = tg.iloc[-1]
        st.caption(
            f"**Why:** New customers (0–12 mo) churn at **{early['ChurnPct']}%** — "
            f"over 3× the rate of long-tenured customers ({late['ChurnPct']}%). "
            "The first year is the highest-risk window; early engagement programs "
            "could significantly reduce this."
        )

    # ── Row 3: TechSupport + SeniorCitizen ────
    col5, col6 = st.columns(2)
    with col5:
        fig, stats = churn_bar("TechSupport", "Churn Rate by Tech Support")
        st.plotly_chart(fig, width="stretch")
        no_sup  = stats[stats["TechSupport"] == "No"]["ChurnPct"].values
        yes_sup = stats[stats["TechSupport"] == "Yes"]["ChurnPct"].values
        if len(no_sup) and len(yes_sup):
            st.caption(
                f"**Why:** Customers without tech support churn at **{no_sup[0]:.1f}%** "
                f"versus {yes_sup[0]:.1f}% with support — a {no_sup[0]-yes_sup[0]:.1f} pp "
                "gap. Unresolved technical issues are a direct driver of cancellations."
            )

    with col6:
        sc = df.copy()
        sc["SeniorCitizenLabel"] = sc["SeniorCitizen"].map({0: "Non-Senior", 1: "Senior"})
        # Compute stats inline (SeniorCitizenLabel is a derived column, not in original df)
        stats2 = (
            sc.groupby("SeniorCitizenLabel")["Churn"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "ChurnRate", "count": "N"})
        )
        stats2["ChurnPct"] = (stats2["ChurnRate"] * 100).round(1)
        fig2 = px.bar(
            stats2, x="SeniorCitizenLabel", y="ChurnPct",
            text=stats2["ChurnPct"].apply(lambda v: f"{v:.1f}%"),
            color="ChurnPct",
            color_continuous_scale=["#22C55E", "#F59E0B", "#EF4444"],
            labels={"ChurnPct": "Churn Rate (%)", "SeniorCitizenLabel": "Customer Segment"},
            title="Churn Rate by Senior Citizen Status",
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            height=350, showlegend=False, coloraxis_showscale=False,
            margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig2, width="stretch")
        sr = stats2[stats2["SeniorCitizenLabel"] == "Senior"]["ChurnPct"].values
        ns = stats2[stats2["SeniorCitizenLabel"] == "Non-Senior"]["ChurnPct"].values
        if len(sr) and len(ns):
            st.caption(
                f"**Why:** Senior citizens churn at **{sr[0]:.1f}%** vs {ns[0]:.1f}% "
                "for non-seniors. This segment may find the service complex or "
                "cost-prohibitive — a simplified plan or senior discount could help."
            )


# ──────────────────────────────────────────────
# PAGE 3 – RISK & ACTIONS
# ──────────────────────────────────────────────
def page_risk_and_actions(df: pd.DataFrame, results: dict):
    st.title("⚠️ Risk & Actions")

    # ── 3A: Model Comparison ──────────────────
    st.subheader("Model Performance Comparison")
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Logistic Regression": [
            results["lr_metrics"]["Accuracy"],
            results["lr_metrics"]["Precision"],
            results["lr_metrics"]["Recall"],
            results["lr_metrics"]["F1"],
            results["lr_metrics"]["ROC-AUC"],
        ],
        "Random Forest": [
            results["rf_metrics"]["Accuracy"],
            results["rf_metrics"]["Precision"],
            results["rf_metrics"]["Recall"],
            results["rf_metrics"]["F1"],
            results["rf_metrics"]["ROC-AUC"],
        ],
    })
    st.dataframe(
        metrics_df.style.highlight_max(subset=["Logistic Regression", "Random Forest"],
                                       color="#D1FAE5", axis=1),
        width="stretch", hide_index=True,
    )
    st.success(
        f"✅ **Best model: {results['best_name']}** "
        f"(ROC-AUC = {results['best_metrics']['ROC-AUC']:.4f})"
    )

    # ── 3B: Confusion Matrix + Feature Importance ──
    cm_col, fi_col = st.columns(2)

    with cm_col:
        st.subheader("Confusion Matrix (Best Model)")
        cm = results["cm"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm[::-1], x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: Yes", "Actual: No"],
            colorscale="Blues",
            text=cm[::-1], texttemplate="%{text}",
            showscale=False,
        ))
        fig_cm.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_cm, width="stretch")

    with fi_col:
        st.subheader(f"Top 15 Feature Importances ({results['best_name']})")
        top15 = results["feat_imp"].head(15)
        fig_fi = px.bar(
            top15[::-1], x="Importance", y="Feature",
            orientation="h", color="Importance",
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(height=320, showlegend=False,
                              coloraxis_showscale=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_fi, width="stretch")

    st.markdown("---")

    # ── 3C: Top-20 High-Risk Customers ────────
    st.subheader("🚨 Top 20 High-Risk Active Customers")
    st.caption(
        "Only customers who have NOT yet churned are listed - these are the "
        "customers a retention team can still act on."
    )
    risk_df = df[["customerID", "Contract", "InternetService",
                  "MonthlyCharges", "tenure", "Churn"]].copy()
    risk_df["ChurnProbability"] = results["all_proba"]
    risk_df = risk_df[risk_df["Churn"] == 0]
    risk_df = risk_df.sort_values("ChurnProbability", ascending=False).head(20)
    risk_df["ChurnProbability"] = risk_df["ChurnProbability"].apply(lambda p: f"{p:.2%}")
    st.dataframe(
        risk_df.drop(columns=["Churn"])
        .reset_index(drop=True),
        width="stretch", hide_index=True,
    )

    st.markdown("---")

    # ── 3D: Single Customer Prediction Form ───
    st.subheader("🔮 Predict Churn for a Single Customer")

    with st.form("predict_form"):
        fc1, fc2, fc3 = st.columns(3)
        tenure_in       = fc1.slider("Tenure (months)", 0, 72, 12)
        monthly_in      = fc2.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        contract_in     = fc3.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        fc4, fc5, fc6 = st.columns(3)
        internet_in     = fc4.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_in      = fc5.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        tech_support_in = fc6.selectbox("Tech Support", ["Yes", "No", "No internet service"])

        fc7, fc8, fc9 = st.columns(3)
        senior_in       = fc7.selectbox("Senior Citizen", ["No", "Yes"])
        partner_in      = fc8.selectbox("Partner", ["Yes", "No"])
        dependents_in   = fc9.selectbox("Dependents", ["No", "Yes"])

        submitted = st.form_submit_button("Predict 🎯")

    if submitted:
        # Build a row matching the training feature set
        sample = {
            "gender":              "Male",
            "SeniorCitizen":       1 if senior_in == "Yes" else 0,
            "Partner":             partner_in,
            "Dependents":          dependents_in,
            "tenure":              tenure_in,
            "PhoneService":        "Yes",
            "MultipleLines":       "No",
            "InternetService":     internet_in,
            "OnlineSecurity":      "No",
            "OnlineBackup":        "No",
            "DeviceProtection":    "No",
            "TechSupport":         tech_support_in,
            "StreamingTV":         "No",
            "StreamingMovies":     "No",
            "Contract":            contract_in,
            "PaperlessBilling":    "Yes",
            "PaymentMethod":       payment_in,
            "MonthlyCharges":      monthly_in,
            "TotalCharges":        monthly_in * tenure_in,
        }
        sample_df = pd.DataFrame([sample])

        # Encode using same label encoders
        les = results["label_encoders"]
        for col, le in les.items():
            if col in sample_df.columns:
                val = str(sample_df[col].values[0])
                if val in le.classes_:
                    sample_df[col] = le.transform([val])
                else:
                    sample_df[col] = le.transform([le.classes_[0]])

        feat_cols = results["feature_cols"]
        X_sample  = sample_df[feat_cols].values

        if results["best_scaled"]:
            X_sample = results["scaler"].transform(X_sample)

        prob = results["best_model"].predict_proba(X_sample)[0][1]

        if prob >= 0.70:
            risk_level = "🔴 HIGH RISK"
            color = "#EF4444"
        elif prob >= 0.40:
            risk_level = "🟡 MEDIUM RISK"
            color = "#F59E0B"
        else:
            risk_level = "🟢 LOW RISK"
            color = "#22C55E"

        st.markdown(
            f"""
            <div style="border:2px solid {color};border-radius:10px;
                        padding:20px;text-align:center;background:#F9FAFB;">
                <h3 style="color:{color};margin:0;">{risk_level}</h3>
                <p style="font-size:22px;font-weight:700;margin:8px 0;">
                    Churn Probability: {prob:.1%}
                </p>
                <p style="color:#6B7280;font-size:13px;margin:0;">
                    Model: {results['best_name']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── 3E: Risks / Opportunities / Actions ───
    st.subheader("📋 Strategic Recommendations")

    # Compute live numbers for recommendations
    # Only customers who are still active (Churn == 0) can be targeted by a retention offer
    active_mask     = (df["Churn"] == 0).values
    high_risk_mtm = df[
        (df["Contract"] == "Month-to-month") &
        (results["all_proba"] >= 0.70) &
        active_mask
    ]
    n_high_risk     = ((results["all_proba"] >= 0.70) & active_mask).sum()
    rev_at_risk_hr  = high_risk_mtm["MonthlyCharges"].sum()
    fiber_churn_pct = df[df["InternetService"] == "Fiber optic"]["Churn"].mean() * 100
    senior_pct      = df[df["SeniorCitizen"] == 1]["Churn"].mean() * 100
    no_tech_pct     = df[df["TechSupport"] == "No"]["Churn"].mean() * 100

    tab_r, tab_o, tab_a = st.tabs(["🔴 Risks", "🟢 Opportunities", "🎯 Recommended Actions"])

    with tab_r:
        st.markdown(f"""
| # | Risk | Evidence |
|---|------|----------|
| 1 | **Month-to-month customer flight** | {(df['Contract']=='Month-to-month').sum():,} customers on flexible contracts; churn rate {df[df['Contract']=='Month-to-month']['Churn'].mean()*100:.1f}% |
| 2 | **Fiber optic price sensitivity** | Fiber customers churn at **{fiber_churn_pct:.1f}%** — the highest of all internet types |
| 3 | **High-risk pipeline** | {n_high_risk:,} active customers have model-predicted churn probability ≥ 70%; monthly revenue exposure of month-to-month customers among them **${rev_at_risk_hr:,.0f}** |
| 4 | **Senior citizen attrition** | {df[df['SeniorCitizen']==1].shape[0]:,} senior customers with {senior_pct:.1f}% churn rate |
| 5 | **Low tech support adoption** | {(df['TechSupport']=='No').sum():,} subscribers without support; {no_tech_pct:.1f}% churn vs {df[df['TechSupport']=='Yes']['Churn'].mean()*100:.1f}% with support |
        """)

    with tab_o:
        long_term = df[df["Contract"] != "Month-to-month"]
        upsell_pool = df[
            (results["all_proba"] < 0.40) &
            active_mask &
            (df["InternetService"] != "No") &
            (df["TechSupport"] == "No")
        ]
        st.markdown(f"""
| # | Opportunity | Potential |
|---|-------------|-----------|
| 1 | **Convert M2M → annual contracts** | {(df['Contract']=='Month-to-month').sum():,} eligible customers; annual contract churn is {df[df['Contract']=='One year']['Churn'].mean()*100:.1f}% |
| 2 | **Tech Support upsell to low-risk customers** | {len(upsell_pool):,} active, low-risk customers with internet but no tech support — low-touch upsell potential |
| 3 | **Loyal customer programmes** | {(df['tenure']>=48).sum():,} customers with 4+ year tenure — high lifetime value, ripe for premium or bundled upgrade |
| 4 | **Senior-friendly simplified plan** | Dedicated plan for {df[df['SeniorCitizen']==1].shape[0]:,} seniors could cut that segment's churn from {senior_pct:.1f}% toward the dataset average of {df['Churn'].mean()*100:.1f}% |
        """)

    with tab_a:
        st.markdown(f"""
### Priority Actions

**1 — Targeted Retention Offer for High-Value, High-Risk M2M Customers**
- **Who:** {len(high_risk_mtm):,} active month-to-month customers with churn probability ≥ 70%
- **Revenue exposure:** ${rev_at_risk_hr:,.0f} / month
- **Action:** Proactively offer a 15–20% discount on an annual contract upgrade, free 3-month tech support trial, or a loyalty reward.
- **Expected impact:** Even a 25% conversion rate saves ~${rev_at_risk_hr * 0.25:,.0f}/month in lost revenue.

**2 — Fiber Optic Satisfaction Programme**
- **Who:** {(df['InternetService']=='Fiber optic').sum():,} fiber customers (churn rate {fiber_churn_pct:.1f}%)
- **Action:** Proactive service-quality outreach, speed upgrade offers, and "price-lock" promotion to 1-year plan.

**3 — New Customer Onboarding (Tenure 0–12 months)**
- First-year churn is the highest risk window.
- **Action:** Assign a dedicated onboarding success contact, automated 30/60/90-day check-in emails, and a first-year loyalty discount.

**4 — Senior Citizen Outreach**
- **Who:** {df[df['SeniorCitizen']==1].shape[0]:,} senior customers at {senior_pct:.1f}% churn
- **Action:** Introduce a "Senior Simplified" plan — reduced complexity, lower price point, priority support line.

**5 — Tech Support Awareness Campaign**
- **Who:** {(df['TechSupport']=='No').sum():,} customers without tech support at {no_tech_pct:.1f}% churn
- **Action:** In-app and email campaign highlighting support benefits; offer first 3 months free to at-risk customers.
        """)


# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
def main():
    # ── Load data and train models ─────────────
    df      = load_data()
    results = train_models(df)

    st.sidebar.title("📡 Telco Churn")
    st.sidebar.markdown("**Customer Churn Analysis\n& Retention Dashboard**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate to",
        ["📊 Executive Overview", "🔎 Churn Drivers", "⚠️ Risk & Actions"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Dataset:** {len(df):,} customers\n\n"
        f"**Best model:** {results['best_name']}\n\n"
        f"**ROC-AUC:** {results['best_metrics']['ROC-AUC']:.4f}"
    )
    st.sidebar.markdown(
        f"[📥 Get Dataset]({KAGGLE_LINK})",
    )

    if page == "📊 Executive Overview":
        page_executive_overview(df, results)
    elif page == "🔎 Churn Drivers":
        page_churn_drivers(df)
    elif page == "⚠️ Risk & Actions":
        page_risk_and_actions(df, results)


if __name__ == "__main__":
    main()
