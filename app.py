"""
Supply Chain Control Tower
--------------------------
Streamlit app for late-delivery monitoring on the DataCo Supply Chain dataset.

What it does:
  * Robust data loading with automatic fallback demo data (no crashes or KeyErrors).
  * Cleans data, drops PII, and calculates Delay and Profit safely.
  * Autopilot Agent: answers questions, highlights root causes, and recommends actions.
  * Overview Dashboard: metrics, late rate by shipping mode, histograms, and trends.
  * Risk Predictor ML Model: trained on order-time features with interactive simulation.
"""

from __future__ import annotations

import html
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# ----------------------------------------------------------------------------
# Config & Styling
# ----------------------------------------------------------------------------
GITHUB_URL = "https://github.com/Raunak4u/supply-chain-control-tower-agent"

FILE_CANDIDATES = [
    "DataCoSupplyChainDataset.csv",
    "Supply Chain Shipment Pricing Dataset.csv",
    "supply_chain.csv",
]
PII_COLUMNS = [
    "Customer Email", "Customer Password", "Customer Fname", "Customer Lname",
    "Customer Street", "Customer Zipcode", "Order Zipcode",
    "Product Description", "Product Image",
]
PROFIT_CANDIDATES = ["Benefit per order", "Order Profit Per Order"]

MODEL_CAT = [
    "Shipping Mode", "Type", "Market", "Order Region",
    "Category Name", "Customer Segment", "Department Name",
]
MODEL_NUM = [
    "Days for shipment (scheduled)", "Order Item Quantity", "Sales",
    "Order Item Discount Rate",
]

INK, MUTED, LINE = "#0F1B2D", "#5B6B80", "#E2E8F0"
BRAND, RISK, OK, WARN = "#0E6BA8", "#D1495B", "#2A9D8F", "#E9A23B"

st.set_page_config(page_title="Supply Chain Control Tower", page_icon="🚚", layout="wide")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, label {{
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
}}
.stApp {{ background: #F3F6FA; color: {INK}; }}
[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {LINE}; }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}
h1, h2, h3, h4, h5 {{ color: {INK}; letter-spacing: -0.01em; }}
.stMarkdown p, .stMarkdown li {{ color: #1F2D3F; }}
.page-title {{ font-size: 34px; font-weight: 800; letter-spacing: -0.02em; margin: 0; color: {INK}; }}
.page-sub {{ color: {MUTED}; margin: 4px 0 20px 0; font-size: 15px; }}
.kpi {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 12px; padding: 16px 18px; height: 100%; }}
.kpi .label {{ font-size: 13px; color: {MUTED}; font-weight: 500; }}
.kpi .value {{ font-size: 28px; font-weight: 700; line-height: 1.25; }}
.kpi .note {{ font-size: 13px; color: {MUTED}; }}
.finding {{ background: #FFFFFF; border: 1px solid {LINE}; border-left: 4px solid var(--c, {BRAND});
    border-radius: 10px; padding: 14px 16px; color: #1F2D3F; line-height: 1.6; margin: 6px 0 10px 0; }}
.briefing-head {{ border-left: 4px solid {BRAND}; padding-left: 12px; margin: 4px 0 14px 0; }}
.briefing-head h3 {{ margin: 0; font-size: 20px; }}
.briefing-head p {{ margin: 2px 0 0 0; color: {MUTED}; font-size: 14px; }}
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Data Loading & Preparation
# ----------------------------------------------------------------------------
def find_default_file() -> str | None:
    here = Path(__file__).resolve().parent
    for folder in (here, here / "data", Path.cwd(), Path.cwd() / "data"):
        for name in FILE_CANDIDATES:
            c = folder / name
            if c.exists():
                return str(c)
    return None


def _read_csv(source) -> pd.DataFrame | None:
    for enc in ("latin-1", "utf-8"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc)
        except Exception:
            continue
    return None


@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(path: str | None, raw_bytes: bytes | None) -> pd.DataFrame:
    df = None
    if raw_bytes is not None:
        df = _read_csv(io.BytesIO(raw_bytes))
    elif path:
        df = _read_csv(path)

    if df is None or df.empty:
        np.random.seed(42)
        n = 180
        real = np.random.randint(1, 7, n)
        sched = np.array([4] * n)
        df = pd.DataFrame({
            'Type': np.random.choice(['DEBIT', 'TRANSFER', 'CASH', 'PAYMENT'], n),
            'Days for shipping (real)': real,
            'Days for shipment (scheduled)': sched,
            'Benefit per order': np.random.uniform(-250, 300, n).round(2),
            'Sales per customer': np.random.uniform(100, 400, n).round(2),
            'Delivery Status': np.random.choice(['Advance shipping', 'Late delivery', 'Shipping on time'], n, p=[0.5, 0.2, 0.3]),
            'Late_delivery_risk': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'Category Name': 'Sporting Goods',
            'Order Region': np.random.choice(['West of USA', 'East of USA', 'Central'], n),
            'Shipping Mode': np.random.choice(['Standard Class', 'First Class', 'Second Class'], n),
            'Market': np.random.choice(['LATAM', 'USCA', 'APAC'], n),
            'Customer Segment': np.random.choice(['Consumer', 'Corporate', 'Home Office'], n),
            'Department Name': 'Fan Shop',
            'Order Item Quantity': np.random.randint(1, 5, n),
            'Sales': np.random.uniform(50, 300, n),
            'Order Item Discount Rate': np.random.uniform(0, 0.2, n),
        })

    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns], errors='ignore')

    if "Late_delivery_risk" not in df.columns:
        if "Days for shipping (real)" in df.columns and "Days for shipment (scheduled)" in df.columns:
            df["Late_delivery_risk"] = (df["Days for shipping (real)"] > df["Days for shipment (scheduled)"]).astype(int)
        else:
            df["Late_delivery_risk"] = 0

    if "Delivery Status" not in df.columns:
        df["Delivery Status"] = np.where(df["Late_delivery_risk"] == 1, "Late delivery", "On time")
    else:
        df = df[df["Delivery Status"] != "Shipping canceled"]

    if "Days for shipping (real)" in df.columns and "Days for shipment (scheduled)" in df.columns:
        df["Delay"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
    else:
        df["Delay"] = 0

    profit_col = next((c for c in PROFIT_CANDIDATES if c in df.columns), None)
    if profit_col:
        df["Profit"] = df[profit_col]
    elif "Benefit per order" in df.columns:
        df["Profit"] = df["Benefit per order"]
    else:
        df["Profit"] = 0.0

    date_col = "order date (DateOrders)"
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df["order_date_parsed"] = dates
        df["Order Month"] = dates.dt.to_period("M").astype(str).where(dates.notna())

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Risk Predictor ML Model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training the risk model...")
def train_model(_df: pd.DataFrame, data_key: str) -> dict:
    cat = [c for c in MODEL_CAT if c in _df.columns]
    num = [c for c in MODEL_NUM if c in _df.columns]
    X, y = _df[cat + num], _df["Late_delivery_risk"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pre = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   encoded_missing_value=np.nan), cat),
            ("num", "passthrough", num),
        ]
    )
    clf = HistGradientBoostingClassifier(
        categorical_features=[True] * len(cat) + [False] * len(num),
        max_iter=150, learning_rate=0.1, random_state=42,
    )
    model = Pipeline([("pre", pre), ("clf", clf)]).fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    defaults = {c: X[c].mode().iloc[0] for c in cat} | {c: float(X[c].median()) for c in num}
    return {
        "model": model, "cat": cat, "num": num, "defaults": defaults,
        "accuracy": accuracy_score(y_te, pred),
        "auc": roc_auc_score(y_te, proba),
        "recall": recall_score(y_te, pred),
        "baseline": max(y_te.mean(), 1 - y_te.mean()),
        "n_test": len(y_te),
    }


# ----------------------------------------------------------------------------
# Helpers & Formatting
# ----------------------------------------------------------------------------
def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{sign}${x / 1e3:.1f}K"
    return f"{sign}${x:,.0f}"


def style_fig(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color=INK, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor=LINE)
    fig.update_yaxes(gridcolor=LINE)
    return fig


def kpi(label: str, value: str, note: str = "", color: str = INK) -> str:
    return (f"<div class='kpi'><div class='label'>{label}</div>"
            f"<div class='value' style='color:{color}'>{value}</div>"
            f"<div class='note'>{note}</div></div>")


def md_to_html(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text))


def rate_table(d: pd.DataFrame, dim: str) -> pd.DataFrame:
    g = d.groupby(dim).agg(
        Orders=("Late_delivery_risk", "size"),
        Late_orders=("Late_delivery_risk", "sum"),
        Late_rate=("Late_delivery_risk", "mean"),
        Profit=("Profit", "sum"),
    ).reset_index()
    return g


def min_volume(d: pd.DataFrame) -> int:
    return max(30, int(len(d) * 0.002))


def show_table(t: pd.DataFrame) -> pd.DataFrame:
    out = t.copy()
    if "Late_rate" in out:
        out["Late_rate"] = (out["Late_rate"] * 100).round(1)
        out = out.rename(columns={"Late_rate": "Late rate (%)"})
    out = out.rename(columns={"Late_orders": "Late orders"})
    if "Profit" in out:
        out["Profit"] = out["Profit"].round(0)
    return out


# ----------------------------------------------------------------------------
# Agent Tools & Logic
# ----------------------------------------------------------------------------
@dataclass
class Result:
    title: str
    finding: str
    color: str = BRAND
    table: pd.DataFrame | None = None
    fig: go.Figure | None = None
    actions: list[str] = field(default_factory=list)


def tool_summary(d: pd.DataFrame) -> Result:
    n, late = len(d), int(d["Late_delivery_risk"].sum())
    late_d = d[d["Late_delivery_risk"] == 1]
    parts = [f"**{late:,} of {n:,} orders ({late / n:.1%})** are late or at risk."]
    if "Shipping Mode" in d:
        t = rate_table(d, "Shipping Mode")
        t = t[t["Orders"] >= min_volume(d)] if (t["Orders"] >= min_volume(d)).any() else t
        worst = t.sort_values("Late_rate", ascending=False).iloc[0]
        parts.append(f"**{worst['Shipping Mode']}** has the highest late rate ({worst['Late_rate']:.1%}).")
    if "Order Region" in d and len(late_d):
        top = late_d["Order Region"].value_counts()
        parts.append(f"**{top.index[0]}** contributes the most late orders ({top.iloc[0]:,}).")
    if len(late_d):
        parts.append(f"Profit tied up in late orders: **{money(late_d['Profit'].sum())}**.")
    return Result("Situation summary", " ".join(parts), BRAND)


def tool_dimension(d: pd.DataFrame, dim: str, label: str, ascending: bool = False) -> Result:
    t = rate_table(d, dim)
    big = t[t["Orders"] >= min_volume(d)]
    big = big if len(big) else t
    pick = big.sort_values("Late_rate", ascending=ascending).iloc[0]
    vol = t.sort_values("Late_orders", ascending=False).iloc[0]
    total_late = max(int(d["Late_delivery_risk"].sum()), 1)
    word = "lowest" if ascending else "highest"
    finding = (
        f"**{pick[dim]}** has the {word} late-delivery rate by {label}: "
        f"**{pick['Late_rate']:.1%}** across {int(pick['Orders']):,} orders. "
        f"By volume, **{vol[dim]}** has the most late orders: {int(vol['Late_orders']):,} "
        f"({vol['Late_orders'] / total_late:.1%} of all late orders)."
    )
    shown = big.sort_values("Late_rate", ascending=ascending).head(10)
    fig = px.bar(
        shown.assign(**{dim: shown[dim].astype(str)}), x="Late_rate", y=dim, orientation="h",
        text=shown["Late_rate"].map("{:.1%}".format),
    )
    fig.update_traces(marker_color=[RISK if i == 0 else BRAND for i in range(len(shown))], textposition="outside")
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(tickformat=".0%", title="Late-delivery rate", range=[0, min(1.0, shown["Late_rate"].max() * 1.2 + 0.02)])

    actions = []
    if dim == "Shipping Mode":
        best = big.sort_values("Late_rate").iloc[0]
        actions.append(f"Re-quote promised delivery days for {pick[dim]} or shift eligible volume to {best[dim]} ({best['Late_rate']:.1%} late rate).")
    else:
        actions.append(f"Add safety stock and carrier checkpoints for {pick[dim]} to cut late orders.")
    return Result(f"Late-delivery risk by {label}", finding, RISK, show_table(shown), fig, actions)


def tool_root_cause(d: pd.DataFrame) -> Result:
    late = d[d["Late_delivery_risk"] == 1]
    if late.empty or "Shipping Mode" not in d or d["Delay"].isna().all():
        return Result("Root-cause analysis", "Not enough shipping data in this view to explain delays.", WARN)
    gap = late["Delay"].mean()
    g = d.groupby("Shipping Mode").agg(
        Orders=("Late_delivery_risk", "size"),
        Late_rate=("Late_delivery_risk", "mean"),
        Late_orders=("Late_delivery_risk", "sum"),
        Promised_days=("Days for shipment (scheduled)", "mean"),
        Actual_days=("Days for shipping (real)", "mean"),
    ).reset_index()
    worst = g.sort_values("Late_rate", ascending=False).iloc[0]
    text = (
        f"Late orders arrive **{gap:.1f} days** after the promised date on average. "
        f"**{worst['Shipping Mode']}** is the weakest link: promises **{worst['Promised_days']:.1f} days** "
        f"but takes **{worst['Actual_days']:.1f}** ({worst['Late_rate']:.0%} late rate)."
    )
    fig = go.Figure()
    fig.add_bar(name="Promised days", x=g["Shipping Mode"], y=g["Promised_days"], marker_color=OK)
    fig.add_bar(name="Actual days", x=g["Shipping Mode"], y=g["Actual_days"], marker_color=RISK)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Average days")
    table = show_table(g.rename(columns={"Promised_days": "Promised days", "Actual_days": "Actual days"}))
    actions = [f"Reset delivery promises for {worst['Shipping Mode']} to match real transit duration."]
    return Result("Why orders are delayed", text, RISK, table, fig, actions)


def tool_financial(d: pd.DataFrame) -> Result:
    late = d[d["Late_delivery_risk"] == 1]
    ontime = d[d["Late_delivery_risk"] == 0]
    if late.empty:
        return Result("Financial exposure", "No late orders in this view.", OK)
    loss_late = late.loc[late["Profit"] < 0, "Profit"].sum()
    n_loss = int((late["Profit"] < 0).sum())
    text = (
        f"Late orders carry **{money(late['Profit'].sum())}** of profit and "
        f"**{n_loss:,}** are loss-making, losing **{money(loss_late)}**. "
        f"Average profit per order: **{money(late['Profit'].mean())}** (late) vs **{money(ontime['Profit'].mean()) if len(ontime) else 'n/a'}** (on-time)."
    )
    fig = go.Figure(go.Bar(
        x=["Late orders", "On-time orders"],
        y=[late["Profit"].mean(), ontime["Profit"].mean() if len(ontime) else 0],
        marker_color=[RISK, OK], text=[money(late["Profit"].mean()), money(ontime["Profit"].mean() if len(ontime) else 0)],
        textposition="outside",
    ))
    fig.update_yaxes(title="Average profit ($)")
    actions = ["Prioritize express handling for high-margin orders."]
    return Result("Financial exposure", text, WARN, None, fig, actions)


def tool_trend(d: pd.DataFrame) -> Result:
    if "order_date_parsed" not in d or d["order_date_parsed"].notna().sum() == 0:
        return Result("Trend over time", "No usable order date column in this dataset view.", WARN)
    dd = d.dropna(subset=["order_date_parsed"])
    m = dd.assign(Period=dd["order_date_parsed"].dt.to_period("M").astype(str)).groupby("Period").agg(
        Orders=("Late_delivery_risk", "size"), Late_rate=("Late_delivery_risk", "mean")
    ).reset_index().sort_values("Period")
    if len(m) < 2:
        return Result("Trend over time", "Not enough temporal span for a trend chart.", WARN)
    fig = px.line(m, x="Period", y="Late_rate", markers=True)
    fig.update_traces(line_color=BRAND)
    fig.update_yaxes(tickformat=".0%", title="Late-delivery rate")
    return Result("Late rate trend", f"Tracked across {len(m)} months.", BRAND, show_table(m), fig, ["Review capacity ahead of seasonal spikes."])


def resolve_dimensions(q: str, cols: set[str]) -> list[tuple[str, str]]:
    dims = []
    if "region" in q: dims.append(("Order Region", "region"))
    if "market" in q: dims.append(("Market", "market"))
    if "categor" in q or "product" in q: dims.append(("Category Name", "category"))
    if "shipping" in q or "mode" in q: dims.append(("Shipping Mode", "shipping mode"))
    if "segment" in q: dims.append(("Customer Segment", "customer segment"))
    return [(c, l) for c, l in dims if c in cols]


def run_agent(question: str, d: pd.DataFrame) -> tuple[list[str], list[Result]]:
    q = question.lower()
    asc = "lowest" in q or "best" in q
    steps, results = [], []
    for col, label in resolve_dimensions(q, set(d.columns)):
        steps.append(f"Ranked risk by {label}")
        results.append(tool_dimension(d, col, label, asc))
    if any(w in q for w in ["why", "reason", "cause", "delay"]):
        steps.append("Analyzed delay root causes")
        results.append(tool_root_cause(d))
    if any(w in q for w in ["profit", "loss", "money", "financial"]):
        steps.append("Calculated financial exposure")
        results.append(tool_financial(d))
    if "trend" in q or "month" in q:
        steps.append("Generated time-series trend")
        results.append(tool_trend(d))
    if not results:
        steps = ["Ran full diagnostic briefing"]
        results = [tool_summary(d), tool_root_cause(d), tool_financial(d)]
    return steps, results


def render_result(res: Result, key: str) -> None:
    st.markdown(f"#### {res.title}")
    st.markdown(f"<div class='finding' style='--c:{res.color}'>{md_to_html(res.finding)}</div>", unsafe_allow_html=True)
    if res.fig is not None:
        st.plotly_chart(style_fig(res.fig, 300), key=f"fig-{key}")
    if res.actions:
        st.markdown("**Recommended actions:**\n" + "\n".join(f"- {a}" for a in res.actions))
    if res.table is not None:
        with st.expander("View table"):
            st.dataframe(res.table, hide_index=True)


# ----------------------------------------------------------------------------
# Sidebar Layout & Filters
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Data Source")
    uploaded = st.file_uploader("Upload DataCo CSV", type=["csv"], help="Drop your CSV here")
    st.markdown(f"[GitHub Repository]({GITHUB_URL})")
    st.markdown("### Filters")
    st.caption("Leave empty to include all orders")

default_path = find_default_file()
full = load_data(default_path, uploaded.getvalue() if uploaded else None)
data_key = f"upload-{uploaded.name}" if uploaded else f"file-{default_path}"

view = full.copy()
with st.sidebar:
    for label, col in (("Market", "Market"), ("Region", "Order Region"), ("Shipping Mode", "Shipping Mode"), ("Customer Segment", "Customer Segment")):
        if col in full.columns:
            opts = sorted(full[col].dropna().unique().tolist())
            chosen = st.multiselect(label, opts, placeholder="All")
            if chosen:
                view = view[view[col].isin(chosen)]

if view.empty:
    st.warning("No orders match your filter criteria. Please adjust filters.")
    st.stop()


# ----------------------------------------------------------------------------
# Header & Top Metrics KPIs
# ----------------------------------------------------------------------------
st.markdown("<p class='page-title'>Supply Chain Control Tower</p>", unsafe_allow_html=True)
st.markdown(f"<p class='page-sub'>Late-delivery monitoring for {len(view):,} orders, with an AI agent that analyzes delays and risks.</p>", unsafe_allow_html=True)

n = len(view)
late_df = view[view["Late_delivery_risk"] == 1] if "Late_delivery_risk" in view.columns else view.iloc[0:0]
n_late = len(late_df)

avg_late_days = float(late_df["Delay"].mean()) if n_late > 0 and "Delay" in late_df.columns else float('nan')
avg_str = f"{avg_late_days:.1f} days" if not np.isnan(avg_late_days) else "n/a"

k1, k2, k3, k4, k5 = st.columns(5)
k1.markdown(kpi("Total Orders", f"{n:,}", "Filtered orders"), unsafe_allow_html=True)
k2.markdown(kpi("Late-Delivery Rate", f"{n_late/n:.1%}" if n else "0%", f"{n_late:,} late orders", RISK), unsafe_allow_html=True)
k3.markdown(kpi("On-Time / Early", f"{n-n_late:,}", f"{(n-n_late)/n:.1%} on-time", OK), unsafe_allow_html=True)
k4.markdown(kpi("Avg Days Late", avg_str, "Late orders only"), unsafe_allow_html=True)
k5.markdown(kpi("Profit on Late Orders", money(late_df['Profit'].sum() if 'Profit' in late_df else 0), f"{int((late_df['Profit'] < 0).sum()) if 'Profit' in late_df else 0} loss-making", WARN), unsafe_allow_html=True)

st.write("")


# ----------------------------------------------------------------------------
# Tabs: Agent | Overview | Risk Predictor
# ----------------------------------------------------------------------------
tab_agent, tab_overview, tab_model = st.tabs(["Agent & Briefing", "Overview Dashboard", "Order Risk Predictor"])

with tab_agent:
    st.markdown("<div class='briefing-head'><h3>Autopilot Briefing</h3><p>Automated breakdown of current supply chain performance.</p></div>", unsafe_allow_html=True)
    for idx, res in enumerate([tool_summary(view), tool_root_cause(view), tool_financial(view)]):
        render_result(res, f"auto-{idx}")

    st.divider()
    st.markdown("### Ask the Agent")
    examples = ["Which region has the highest risk?", "Why are orders delayed?", "Show financial exposure", "Show trend"]
    ex_cols = st.columns(len(examples))
    for c, ex in zip(ex_cols, examples):
        c.button(ex, key=f"ex-{ex}", on_click=lambda e=ex: st.session_state.update(agent_q=e))

    question = st.text_input("Ask a question about your supply chain", key="agent_q", placeholder="e.g. Which shipping mode is worst?")
    if question.strip():
        steps, results = run_agent(question, view)
        with st.expander("Agent Reasoning Steps"):
            for s_idx, step in enumerate(steps, 1):
                st.write(f"{s_idx}. {step}")
        for r_idx, res in enumerate(results):
            render_result(res, f"q-{r_idx}")

with tab_overview:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Late Rate by Shipping Mode")
        if "Shipping Mode" in view:
            t = rate_table(view, "Shipping Mode").sort_values("Late_rate", ascending=False)
            fig = px.bar(t, x="Shipping Mode", y="Late_rate", text=t["Late_rate"].map("{:.0%}".format))
            fig.update_traces(marker_color=[RISK if i == 0 else BRAND for i in range(len(t))], textposition="outside")
            fig.update_yaxes(tickformat=".0%", title=None)
            st.plotly_chart(style_fig(fig), key="ov-mode")
    with col2:
        st.markdown("#### Delivery Delay Distribution")
        if "Delay" in view and view["Delay"].notna().any():
            fig = px.histogram(view, x="Delay", color="Delivery Status", nbins=12, color_discrete_sequence=[RISK, OK, BRAND, WARN])
            fig.update_xaxes(title="Actual minus scheduled days")
            st.plotly_chart(style_fig(fig), key="ov-delay")

    st.markdown("#### Raw Order Explorer")
    st.dataframe(view.head(100), use_container_width=True)

with tab_model:
    st.markdown("### Order-Time Late Risk Predictor")
    st.markdown("Machine learning model trained on variables known **at the moment of order placement**.")
    
    model_res = train_model(full, data_key)
    
    m_cols = st.columns(4)
    m_cols[0].markdown(kpi("Test Accuracy", f"{model_res['accuracy']:.1%}", f"Baseline: {model_res['baseline']:.1%}"), unsafe_allow_html=True)
    m_cols[1].markdown(kpi("ROC-AUC", f"{model_res['auc']:.3f}", "Discrimination power", BRAND), unsafe_allow_html=True)
    m_cols[2].markdown(kpi("Recall", f"{model_res['recall']:.1%}", "Late orders caught", WARN), unsafe_allow_html=True)
    m_cols[3].markdown(kpi("Test Split", f"{model_res['n_test']:,} orders", "Held-out set"), unsafe_allow_html=True)
    
    st.write("")
    st.markdown("#### Simulate a New Order")
    with st.form("risk_sim_form"):
        f_cols = st.columns(2)
        sim_inputs = {}
        with f_cols[0]:
            for col in model_res["cat"]:
                opts = sorted(full[col].dropna().unique().tolist())
                default_val = model_res["defaults"][col]
                default_idx = opts.index(default_val) if default_val in opts else 0
                sim_inputs[col] = st.selectbox(col, opts, index=default_idx)
        with f_cols[1]:
            for col in model_res["num"]:
                default_val = float(model_res["defaults"][col])
                sim_inputs[col] = st.number_input(col, value=default_val)
        
        if st.form_submit_button("Calculate Risk Score"):
            sim_df = pd.DataFrame([sim_inputs])
            prob = model_res["model"].predict_proba(sim_df)[0, 1]
            if prob >= 0.5:
                st.error(f"⚠️ **High Late Delivery Risk ({prob:.1%})** — Recommended to upgrade shipping mode or flag for expedition.")
            else:
                st.success(f"✅ **Low Risk ({prob:.1%})** — Projected to arrive on time.")

if uploaded is None and default_path is None:
    st.info("ℹ️ Running on demo data (180 mock orders). Upload your DataCo CSV from the sidebar for full analysis.")
else:
    st.success(f"Successfully loaded dataset: {len(view):,} orders active.")