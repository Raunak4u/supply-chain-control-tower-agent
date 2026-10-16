"""
Supply Chain Control Tower
--------------------------
Streamlit app for late-delivery monitoring on the DataCo Supply Chain dataset.

What t does
  * Cleans the data (drops PII columns and cancelled orders, builds Delay and Profit).
  * Shows KPIs, charts and a high-risk order table with sidebar filters.
  * Autopilot agent: writes a briefing automatically and answers questions by
    choosing analysis tools, running them on the filtered data and returning
    findings plus recommended actions.
  * Real risk model: trained on information known at order time, scored on a
    held-out 20% split. No hard-coded accuracy numbers.

Run:  streamlit run app.py
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
# Config
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
# Data loading and preparation
# ----------------------------------------------------------------------------
def find_default_file() -> str | None:
    here = Path(__file__).resolve().parent
    for folder in (here, here / "data", Path.cwd(), Path.cwd() / "data"):
        for name in FILE_CANDIDATES:
            candidate = folder / name
            if candidate.exists():
                return str(candidate)
    return None


def _read_csv(source) -> pd.DataFrame:
    for enc in ("latin-1", "utf-8"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode the CSV file.")


@st.cache_data(show_spinner="Loading and cleaning data...")
def load_data(path: str | None, raw: bytes | None) -> pd.DataFrame:
    if raw is not None:
        df = _read_csv(io.BytesIO(raw))
    elif path:
        df = _read_csv(path)
    else:
        # Fallback demo data so the app runs smoothly out-of-the-box
        np.random.seed(42)
        n = 180
        return pd.DataFrame({
            'Type': np.random.choice(['DEBIT', 'TRANSFER', 'CASH', 'PAYMENT'], n),
            'Days for shipping (real)': np.random.randint(1, 7, n),
            'Days for shipment (scheduled)': 4,
            'Benefit per order': np.random.uniform(-250, 300, n).round(2),
            'Sales per customer': np.random.uniform(100, 400, n).round(2),
            'Delivery Status': np.random.choice(['Advance shipping', 'Late delivery', 'Shipping on time'], n, p=[0.5, 0.2, 0.3]),
            'Late_delivery_risk': np.random.choice([0, 1], n, p=[0.8, 0.2]),
            'Category Name': 'Sporting Goods',
            'Order Region': 'West of USA',
            'Shipping Mode': 'Standard Class',
            'Market': 'LATAM',
            'Profit': np.random.uniform(-100, 300, n),
            'Order Item Quantity': np.random.randint(1, 5, n),
            'Sales': np.random.uniform(50, 300, n),
            'Order Item Discount Rate': np.random.uniform(0, 0.2, n)
        })

    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])

    real, sched = "Days for shipping (real)", "Days for shipment (scheduled)"
    if "Late_delivery_risk" not in df.columns:
        if real in df.columns and sched in df.columns:
            df["Late_delivery_risk"] = (df[real] > df[sched]).astype(int)
        else:
            df["Late_delivery_risk"] = 0

    if "Delivery Status" not in df.columns:
        df["Delivery Status"] = np.where(df["Late_delivery_risk"] == 1, "Late delivery", "On time")
    else:
        df = df[df["Delivery Status"] != "Shipping canceled"]

    df["Delay"] = (df[real] - df[sched]) if real in df.columns and sched in df.columns else 0

    profit_col = next((c for c in PROFIT_CANDIDATES if c in df.columns), None)
    df["Profit"] = df[profit_col] if profit_col else df.get("Benefit per order", 0.0)

    date_col = "order date (DateOrders)"
    if date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df["order_date_parsed"] = dates
        df["Order Month"] = dates.dt.to_period("M").astype(str).where(dates.notna())

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Risk model (trained on order-time information only, so no target leakage)
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
# Helpers
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
# Agent: tools, router and result renderer
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
    fig.update_traces(marker_color=[RISK if i == 0 else BRAND for i in range(len(shown))],
                      textposition="outside")
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(tickformat=".0%", title="Late-delivery rate", range=[0, min(1.0, shown["Late_rate"].max() * 1.2 + 0.02)])

    actions = []
    if dim == "Shipping Mode":
        best = big.sort_values("Late_rate").iloc[0]
        actions.append(f"Re-quote promised delivery days for {pick[dim]} or shift eligible volume to "
                       f"{best[dim]} (late rate {best['Late_rate']:.1%}).")
    elif dim in ("Category Name", "Department Name"):
        actions.append(f"Audit packing and supplier lead time for {pick[dim]} and add priority handling.")
    elif dim in ("Customer Segment", "Type"):
        actions.append(f"Track {pick[dim]} separately in weekly reviews; set an alert at its current late rate.")
    else:
        actions.append(f"Add safety stock and a carrier checkpoint for {pick[dim]} to cut late orders.")
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
    g = g[g["Orders"] >= min_volume(d)] if (g["Orders"] >= min_volume(d)).any() else g
    worst = g.sort_values("Late_rate", ascending=False).iloc[0]
    share = worst["Late_orders"] / max(int(late.shape[0]), 1)
    text = (
        f"Late orders arrive **{gap:.1f} days** after the promised date on average. "
        f"**{worst['Shipping Mode']}** is the weakest link: it promises **{worst['Promised_days']:.1f} days** "
        f"but takes **{worst['Actual_days']:.1f}**, giving a {worst['Late_rate']:.0%} late rate "
        f"and {share:.0%} of all late orders."
    )
    sched = d.groupby("Days for shipment (scheduled)")["Late_delivery_risk"].agg(["mean", "size"])
    sched = sched[sched["size"] >= min_volume(d)]
    if len(sched):
        k = sched["mean"].idxmax()
        text += f" Orders promised in **{int(k)} day(s)** are late {sched.loc[k, 'mean']:.0%} of the time."
    fig = go.Figure()
    fig.add_bar(name="Promised days", x=g["Shipping Mode"], y=g["Promised_days"], marker_color=OK)
    fig.add_bar(name="Actual days", x=g["Shipping Mode"], y=g["Actual_days"], marker_color=RISK)
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Average days")
    table = g.rename(columns={"Promised_days": "Promised days", "Actual_days": "Actual days"})
    table = show_table(table).round({"Promised days": 2, "Actual days": 2})
    actions = [
        f"Reset the promise for {worst['Shipping Mode']} to match real transit time, or fix the carrier handoff.",
        "Add an auto-rule: when scheduled days are 2 or fewer, flag the order for early carrier follow-up.",
    ]
    return Result("Why orders are delayed", text, RISK, table, fig, actions)


def tool_financial(d: pd.DataFrame) -> Result:
    late = d[d["Late_delivery_risk"] == 1]
    ontime = d[d["Late_delivery_risk"] == 0]
    if late.empty:
        return Result("Financial exposure", "No late orders in this view.", OK)
    loss_late = late.loc[late["Profit"] < 0, "Profit"].sum()
    loss_all = d.loc[d["Profit"] < 0, "Profit"].sum()
    n_loss = int((late["Profit"] < 0).sum())
    share = loss_late / loss_all if loss_all else 0
    q75 = d["Profit"].quantile(0.75)
    text = (
        f"Late orders carry **{money(late['Profit'].sum())}** of profit and "
        f"**{n_loss:,}** of them are loss-making, losing **{money(loss_late)}**. "
        f"That is **{share:.0%}** of all losses in this view. Average profit per order is "
        f"**{money(late['Profit'].mean())}** for late orders vs **{money(ontime['Profit'].mean()) if len(ontime) else 'n/a'}** on time."
    )
    fig = go.Figure(go.Bar(
        x=["Late orders", "On-time orders"],
        y=[late["Profit"].mean(), ontime["Profit"].mean() if len(ontime) else 0],
        marker_color=[RISK, OK], text=[money(late["Profit"].mean()), money(ontime["Profit"].mean() if len(ontime) else 0)],
        textposition="outside",
    ))
    fig.update_yaxes(title="Average profit per order ($)")
    actions = [
        f"Give priority shipping to high-margin orders (profit above {money(q75)}, the top quartile).",
        "Review loss-making late orders for discount and shipping-cost leakage.",
    ]
    return Result("Financial exposure", text, WARN, None, fig, actions)


def tool_trend(d: pd.DataFrame) -> Result:
    if "order_date_parsed" not in d or d["order_date_parsed"].notna().sum() == 0:
        return Result("Trend over time", "This dataset has no usable order date column, "
                                         "so a trend cannot be shown.", WARN)
    dd = d.dropna(subset=["order_date_parsed"])
    first, last = dd["order_date_parsed"].min(), dd["order_date_parsed"].max()
    span_days = (last - first).days
    n_months = dd["order_date_parsed"].dt.to_period("M").nunique()

    if n_months >= 3:
        code, unit = "M", "month"
    elif span_days >= 14:
        code, unit = "W", "week"
    else:
        code, unit = "D", "day"

    period = dd["order_date_parsed"].dt.to_period(code)
    m = (dd.assign(Period=period.astype(str))
            .groupby("Period")
            .agg(Orders=("Late_delivery_risk", "size"), Late_rate=("Late_delivery_risk", "mean"))
            .reset_index().sort_values("Period"))
    m = m[m["Orders"] >= 10]
    span = f"{first:%d %b %Y} to {last:%d %b %Y}"
    if len(m) < 2:
        return Result("Trend over time",
                      f"Orders in this view span {span}, which is too short for a trend. "
                      "Clear some filters, or check that the CSV is the full dataset.", WARN)

    hi, lo = m.loc[m["Late_rate"].idxmax()], m.loc[m["Late_rate"].idxmin()]
    change = m["Late_rate"].iloc[-1] - m["Late_rate"].iloc[0]
    text = (
        f"Across {len(m)} {unit}s ({span}), late rate peaked in **{hi['Period']}** at "
        f"**{hi['Late_rate']:.1%}** and was lowest in **{lo['Period']}** at **{lo['Late_rate']:.1%}**. "
        f"From the first to the last {unit} it moved **{change * 100:+.1f} percentage points**."
    )
    fig = px.line(m, x="Period", y="Late_rate", markers=True)
    fig.update_traces(line_color=BRAND)
    fig.update_yaxes(tickformat=".0%", title="Late-delivery rate")
    fig.update_xaxes(title=None)
    return Result(f"Late rate by {unit}", text, BRAND, show_table(m.tail(12)), fig,
                  ["Plan carrier capacity and buffer stock ahead of the peak periods."])


def tool_recommend(d: pd.DataFrame) -> Result:
    rows = []
    min_n = min_volume(d)
    if "Shipping Mode" in d:
        t = rate_table(d, "Shipping Mode")
        t = t[t["Orders"] >= min_n]
        if len(t) >= 2:
            worst, best = t.sort_values("Late_rate").iloc[-1], t.sort_values("Late_rate").iloc[0]
            gain = int(worst["Orders"] * (worst["Late_rate"] - best["Late_rate"]))
            rows.append(("Fix the promise or carrier for " + str(worst["Shipping Mode"]),
                         f"Late rate {worst['Late_rate']:.1%} vs {best['Late_rate']:.1%} for {best['Shipping Mode']}",
                         f"Up to ~{gain:,} fewer late orders if it matched {best['Shipping Mode']} (scenario)"))
    late = d[d["Late_delivery_risk"] == 1]
    for col, label in (("Order Region", "region"), ("Order City", "city"), ("Category Name", "category")):
        if col in d and len(late):
            top = late[col].value_counts()
            rows.append((f"Prioritise {top.index[0]} ({label})",
                         f"{top.iloc[0]:,} late orders, {top.iloc[0] / len(late):.1%} of all late orders",
                         "Targets the largest single pool of late orders"))
    if not rows:
        return Result("Recommended actions", "Not enough data in this view to rank actions.", WARN)
    table = pd.DataFrame(rows, columns=["Action", "Evidence", "Expected impact"])
    return Result("Recommended actions", f"Top priority: **{rows[0][0]}**. {rows[0][1]}.", OK, table, None,
                  [r[0] for r in rows[:3]])


def resolve_dimensions(q: str, cols: set[str]) -> list[tuple[str, str]]:
    dims: list[tuple[str, str]] = []

    def has(*words: str) -> bool:
        return any(re.search(rf"\b{w}", q) for w in words)

    if has("city", "cities"):
        col = "Customer City" if ("customer" in q and "Customer City" in cols) else (
            "Order City" if "Order City" in cols else "Customer City")
        dims.append((col, col.lower()))
    if has("state"):
        col = "Customer State" if "Customer State" in cols else "Order State"
        dims.append((col, "state"))
    if has("country", "countries"):
        col = "Customer Country" if ("customer" in q and "Customer Country" in cols) else "Order Country"
        dims.append((col, "country"))
    if has("region"):
        dims.append(("Order Region", "region"))
    if has("market"):
        dims.append(("Market", "market"))
    if has("categor", "product", "sporting", "goods"):
        dims.append(("Category Name", "category"))
    if has("department"):
        dims.append(("Department Name", "department"))
    if has("segment"):
        dims.append(("Customer Segment", "customer segment"))
    if has("shipping", "mode", "class", "carrier"):
        dims.append(("Shipping Mode", "shipping mode"))
    if has("payment"):
        dims.append(("Type", "payment type"))
    return [(c, l) for c, l in dims if c in cols]


def run_agent(question: str, d: pd.DataFrame) -> tuple[list[str], list[Result]]:
    q = question.lower()
    asc = bool(re.search(r"\b(lowest|least|best|safest|minimum|min)\b", q))
    steps: list[str] = []
    results: list[Result] = []

    for col, label in resolve_dimensions(q, set(d.columns)):
        steps.append(f"Ranked late-delivery risk by {label}")
        results.append(tool_dimension(d, col, label, asc))
    if re.search(r"\b(why|reason|cause|root|bottleneck|delayed|delays?)\b", q):
        steps.append("Compared promised vs actual days to find the root cause")
        results.append(tool_root_cause(d))
    if re.search(r"\b(profit|loss|benefit|money|financial|exposure|revenue|cost)\b", q):
        steps.append("Calculated financial exposure of late orders")
        results.append(tool_financial(d))
    if re.search(r"\b(trend|month|monthly|time|over time|seasonal)\b", q):
        steps.append("Built the monthly late-rate trend")
        results.append(tool_trend(d))
    if re.search(r"\b(action|suggest|recommend|mitigat|fix|improve|reduce|solve|plan)\w*", q):
        steps.append("Ranked actions by evidence and expected impact")
        results.append(tool_recommend(d))
    if re.search(r"\b(summary|overview|status|brief|health)\b", q):
        steps.append("Summarised the current situation")
        results.insert(0, tool_summary(d))

    if not results:
        steps = ["No specific topic detected, so I ran the full briefing"]
        results = [tool_summary(d), tool_root_cause(d), tool_financial(d), tool_recommend(d)]
    return steps, results


def render_result(res: Result, key: str) -> None:
    st.markdown(f"#### {res.title}")
    st.markdown(f"<div class='finding' style='--c:{res.color}'>{md_to_html(res.finding)}</div>",
                unsafe_allow_html=True)
    if res.fig is not None:
        st.plotly_chart(style_fig(res.fig, 300), key=f"fig-{key}")
    if res.actions:
        st.markdown("**Recommended actions**\n" + "\n".join(f"- {a}" for a in res.actions))
    if res.table is not None:
        with st.expander("View data"):
            st.dataframe(res.table, hide_index=True)


# ----------------------------------------------------------------------------
# Sidebar: data source and filters
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Data")
    uploaded = st.file_uploader("Upload the DataCo CSV", type="csv",
                                help="Only needed if the file is not next to app.py.")
    st.markdown(f"[GitHub repository]({GITHUB_URL})")

try:
    if uploaded is not None:
        full = load_data(None, uploaded.getvalue())
        data_key = f"upload-{uploaded.name}-{uploaded.size}"
    else:
        default_path = find_default_file()
        if default_path:
            full = load_data(default_path, None)
            data_key = f"file-{default_path}"
        else:
            full = load_data(None, None)
            data_key = "demo-180"
except Exception as exc:
    st.warning(f"Using demo data because: {exc}")
    full = load_data(None, None)
    data_key = "demo-fallback"

view = full
with st.sidebar:
    st.markdown("### Filters")
    st.caption("Leave a filter empty to include everything.")
    for label, col in (("Market", "Market"), ("Region", "Order Region"),
                       ("Shipping mode", "Shipping Mode"), ("Customer segment", "Customer Segment")):
        if col in full.columns:
            chosen = st.multiselect(label, sorted(full[col].dropna().unique().tolist()), placeholder="All")
            if chosen:
                view = view[view[col].isin(chosen)]

if view.empty:
    st.warning("No orders match these filters. Clear a filter to continue.")
    st.stop()

# ----------------------------------------------------------------------------
# Header and KPIs
# ----------------------------------------------------------------------------
st.markdown("<p class='page-title'>Supply chain control tower</p>", unsafe_allow_html=True)
st.markdown(
    f"<p class='page-sub'>Late-delivery monitoring for {len(view):,} orders, with an agent that "
    f"explains delays and recommends actions.</p>", unsafe_allow_html=True)

n = len(view)
late_df = view[view["Late_delivery_risk"] == 1]
n_late = len(late_df)
avg_late_days = late_df["Delay"].mean() if n_late and view["Delay"].notna().any() else np.nan
k = st.columns(5)
k[0].markdown(kpi("Total orders", f"{n:,}", "After removing cancelled orders"), unsafe_allow_html=True)
k[1].markdown(kpi("Late-delivery rate", f"{n_late / n:.1%}", f"{n_late:,} late orders", RISK), unsafe_allow_html=True)
k[2].markdown(kpi("On-time or early", f"{n - n_late:,}", f"{(n - n_late) / n:.1%} of orders", OK), unsafe_allow_html=True)
k[3].markdown(kpi("Avg days late", "n/a" if np.isnan(avg_late_days) else f"{avg_late_days:.1f}", "Late orders only"),
              unsafe_allow_html=True)
k[4].markdown(kpi("Profit on late orders", money(late_df["Profit"].sum() if 'Profit' in late_df else 0),
                  f"{int((late_df['Profit'] < 0).sum()) if 'Profit' in late_df else 0:,} loss-making"), unsafe_allow_html=True)
st.write("")

tab_agent, tab_overview, tab_model = st.tabs(["Agent", "Overview", "Risk predictor"])

# ----------------------------------------------------------------------------
# Tab 1: Agent
# ----------------------------------------------------------------------------
with tab_agent:
    st.markdown(
        "<div class='briefing-head'><h3>Autopilot briefing</h3>"
        "<p>Generated automatically from the filtered orders. It updates when you change a filter.</p></div>",
        unsafe_allow_html=True)
    for i, res in enumerate([tool_summary(view), tool_root_cause(view), tool_recommend(view)]):
        render_result(res, f"auto-{i}")

    st.divider()
    st.markdown("### Ask the agent")
    st.caption("Ask in plain English. The agent picks the analysis tools, runs them and explains the result.")

    examples = [
        "Which customer city has the highest late delivery risk?",
        "Why are orders delayed?",
        "How much profit is at risk?",
        "Which category is worst and what should we do?",
        "Show the monthly trend",
    ]
    cols = st.columns(len(examples))
    for c, ex in zip(cols, examples):
        c.button(ex, key=f"ex-{ex}", on_click=lambda e=ex: st.session_state.update(agent_q=e))

    question = st.text_input("Your question", key="agent_q",
                            placeholder="e.g. Which region has the most late orders? Suggest actions.")
    if question.strip():
        steps, results = run_agent(question, view)
        with st.expander("What the agent did", expanded=False):
            st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)))
        for i, res in enumerate(results):
            render_result(res, f"q-{i}")

# ----------------------------------------------------------------------------
# Tab 2: Overview
# ----------------------------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Late rate by shipping mode")
        if "Shipping Mode" in view:
            t = rate_table(view, "Shipping Mode").sort_values("Late_rate", ascending=False)
            fig = px.bar(t, x="Shipping Mode", y="Late_rate", text=t["Late_rate"].map("{:.0%}".format))
            fig.update_traces(marker_color=[RISK if i == 0 else BRAND for i in range(len(t))], textposition="outside")
            fig.add_hline(y=n_late / n if n else 0, line_dash="dot", line_color=MUTED,
                          annotation_text="Overall", annotation_position="top left")
            fig.update_yaxes(tickformat=".0%", title=None, range=[0, min(1.05, t["Late_rate"].max() * 1.2)])
            fig.update_xaxes(title=None)
            st.plotly_chart(style_fig(fig), key="ov-mode")
    with c2:
        st.markdown("#### Days late or early")
        if "Delay" in view and view["Delay"].notna().any():
            fig = px.histogram(view, x="Delay", color="Delivery Status", nbins=12,
                               color_discrete_sequence=[RISK, OK, BRAND, WARN])
            fig.update_layout(bargap=0.05)
            fig.update_xaxes(title="Actual minus promised days")
            fig.update_yaxes(title="Orders")
            st.plotly_chart(style_fig(fig), key="ov-delay")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Late rate over time")
        trend = tool_trend(view)
        if trend.fig is not None:
            st.plotly_chart(style_fig(trend.fig), key="ov-trend")
        else:
            st.info(trend.finding)
    with c4:
        st.markdown("#### Regions with the most late orders")
        if "Order Region" in view:
            t = rate_table(view, "Order Region").sort_values("Late_orders", ascending=False).head(10)
            fig = px.bar(t, x="Late_orders", y="Order Region", orientation="h", text="Late_orders")
            fig.update_traces(marker_color=BRAND, textposition="outside")
            fig.update_yaxes(autorange="reversed", title=None)
            fig.update_xaxes(title="Late orders count")
            st.plotly_chart(style_fig(fig), key="ov-region")

# ----------------------------------------------------------------------------
# Tab 3: Risk Predictor
# ----------------------------------------------------------------------------
with tab_model:
    st.markdown("### Order-Time Risk Model")
    st.markdown("Predicts the likelihood of late delivery using features available **at the time the order is placed** (no target leakage).")
    
    model_res = train_model(full, data_key)
    
    m_cols = st.columns(4)
    m_cols[0].markdown(kpi("Test Accuracy", f"{model_res['accuracy']:.1%}", f"Baseline: {model_res['baseline']:.1%}"), unsafe_allow_html=True)
    m_cols[1].markdown(kpi("ROC-AUC", f"{model_res['auc']:.3f}", "Discrimination power", BRAND), unsafe_allow_html=True)
    m_cols[2].markdown(kpi("Recall", f"{model_res['recall']:.1%}", "Late orders caught", WARN), unsafe_allow_html=True)
    m_cols[3].markdown(kpi("Test Split", f"{model_res['n_test']:,} orders", "Held-out evaluation"), unsafe_allow_html=True)
    
    st.write("")
    st.markdown("#### Simulate an Order Risk Score")
    
    with st.form("risk_form"):
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
        
        submitted = st.form_submit_button("Calculate Risk Score")
        if submitted:
            sim_df = pd.DataFrame([sim_inputs])
            prob = model_res["model"].predict_proba(sim_df)[0, 1]
            if prob >= 0.5:
                st.error(f"⚠️ **High Late Delivery Risk ({prob:.1%})** — Recommended to switch shipping mode or add buffer stock.")
            else:
                st.success(f"✅ **Low Risk ({prob:.1%})** — Order projected to arrive on time.")

if uploaded is None and data_key.startswith("demo"):
    st.info("ℹ️ Demo data is running. Upload your DataCo CSV from the sidebar whenever you are ready.")
else:
    st.success(f"✅ Active dataset loaded: {len(view):,} orders")