
"""
SUPPLY CHAIN CONTROL TOWER
Delivery Performance, Operational Intelligence & Predictive Analytics

Run:
    pip install streamlit pandas numpy plotly scikit-learn
    streamlit run app.py
"""

from __future__ import annotations

import html
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# FIX HEADING
import streamlit as st
st.markdown('<style>.block-container{padding-top:3rem!important} header{visibility:hidden;height:0px} h1{margin-top:20px!important}</style>', unsafe_allow_html=True)


from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Supply Chain Control Tower",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

GITHUB_URL = (
    "https://github.com/Raunak4u/"
    "supply-chain-control-tower-agent"
)

FILE_CANDIDATES = [
    "DataCoSupplyChainDataset.csv",
    "Supply Chain Shipment Pricing Dataset.csv",
    "supply_chain.csv",
]

PII_COLUMNS = [
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Zipcode",
    "Order Zipcode",
    "Product Description",
    "Product Image",
]

PROFIT_CANDIDATES = [
    "Benefit per order",
    "Order Profit Per Order",
]

MODEL_CAT = [
    "Shipping Mode",
    "Type",
    "Market",
    "Order Region",
    "Category Name",
    "Customer Segment",
    "Department Name",
]

MODEL_NUM = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Sales",
    "Order Item Discount Rate",
]

INK = "#172B4D"
MUTED = "#64748B"
LINE = "#E2E8F0"
BRAND = "#2563EB"
RISK = "#DC4C5A"
OK = "#159A80"
WARN = "#D99726"


# ============================================================
# 2. PROFESSIONAL CSS
# ============================================================

st.markdown(
    f"""
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    :root {{
        --ink: {INK};
        --muted: {MUTED};
        --line: {LINE};
        --brand: {BRAND};
        --risk: {RISK};
        --ok: {OK};
        --warn: {WARN};
    }}

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background: #F5F7FB;
        color: var(--ink);
    }}

    [data-testid="stHeader"] {{
        background: rgba(245,247,251,.92);
    }}

    [data-testid="stSidebar"] {{
        background: #FFFFFF;
        border-right: 1px solid var(--line);
    }}

    .block-container {{
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }}

    h1, h2, h3, h4 {{
        color: var(--ink);
        letter-spacing: -.035em;
        font-weight: 700;
    }}

    .hero {{
        background: linear-gradient(120deg,#10243A,#174A83);
        border: 1px solid #234B77;
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 24px;
    }}

    .hero-eyebrow {{
        color: #BFDBFE;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}

    .hero-title {{
        color: #FFFFFF;
        font-size: clamp(26px,3vw,36px);
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: -1px;
    }}

    .hero-subtitle {{
        color: #DCE8F7;
        font-size: 14px;
        line-height: 1.8;
        margin-top: 10px;
        max-width: 850px;
    }}

    .section-heading {{
        color: var(--ink);
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -.5px;
        margin: 8px 0 4px 0;
    }}

    .section-caption {{
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 16px;
    }}

    .kpi {{
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 18px;
        min-height: 128px;
        box-shadow: 0 2px 6px rgba(23,43,77,.025);
    }}

    .kpi .label {{
        color: var(--muted);
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }}

    .kpi .value {{
        font-size: clamp(21px,2vw,29px);
        font-weight: 800;
        line-height: 1.25;
        letter-spacing: -.7px;
        overflow-wrap: anywhere;
    }}

    .kpi .note {{
        color: var(--muted);
        font-size: 11px;
        margin-top: 8px;
    }}

    .finding {{
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-left: 4px solid var(--c,var(--brand));
        border-radius: 10px;
        padding: 15px 17px;
        line-height: 1.85;
        font-size: 13px;
        margin: 8px 0 14px 0;
    }}

    .briefing-head {{
        border-left: 4px solid var(--brand);
        padding-left: 14px;
        margin: 6px 0 18px 0;
    }}

    .briefing-head h3 {{
        font-size: 21px;
        margin: 0;
    }}

    .briefing-head p {{
        color: var(--muted);
        font-size: 13px;
        margin: 5px 0 0 0;
    }}

    div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 1px solid var(--line);
    }}

    div[data-testid="stTabs"] button[data-baseweb="tab"] {{
        font-weight: 650;
        padding: 12px 15px;
    }}

    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: var(--brand);
    }}

    div[data-testid="stButton"] > button {{
        border-radius: 9px;
        font-weight: 600;
        border-color: #D7E0EC;
    }}

    div[data-testid="stFormSubmitButton"] > button {{
        background: var(--brand);
        color: white;
        border-radius: 9px;
        min-height: 44px;
        font-weight: 700;
        border: none;
    }}

    div[data-testid="stFormSubmitButton"] > button:hover {{
        background: #1D4ED8;
        color: white;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
    }}

    div[data-testid="stExpander"] {{
        border: 1px solid var(--line);
        border-radius: 10px;
        background: #FFFFFF;
    }}

    hr {{
        border-color: var(--line);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATA LOADING
# ============================================================

def find_default_file() -> str | None:
    """Find a dataset in the app directory or its data folder."""
    here = Path(__file__).resolve().parent

    for folder in (
        here,
        here / "data",
        Path.cwd(),
        Path.cwd() / "data",
    ):
        for filename in FILE_CANDIDATES:
            candidate = folder / filename
            if candidate.is_file():
                return str(candidate)

    return None


def read_csv_safely(source) -> pd.DataFrame | None:
    """Read CSV data using common encodings."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)

            return pd.read_csv(source, encoding=encoding)
        except (UnicodeError, pd.errors.ParserError, OSError, ValueError):
            continue

    return None


def create_demo_data(n: int = 400) -> pd.DataFrame:
    """Create illustrative demo data; not real operational data."""
    rng = np.random.default_rng(42)

    scheduled = rng.integers(2, 7, n)
    actual = np.maximum(
        1,
        scheduled + rng.choice(
            [-2, -1, 0, 0, 1, 2, 3],
            n,
            p=[.08, .17, .20, .15, .18, .14, .08],
        ),
    )

    late = (actual > scheduled).astype(int)

    return pd.DataFrame({
        "Type": rng.choice(
            ["DEBIT", "TRANSFER", "CASH", "PAYMENT"], n
        ),
        "Days for shipping (real)": actual,
        "Days for shipment (scheduled)": scheduled,
        "Benefit per order": rng.normal(35, 85, n).round(2),
        "Sales per customer": rng.uniform(50, 500, n).round(2),
        "Sales": rng.uniform(50, 500, n).round(2),
        "Delivery Status": np.where(
            late == 1, "Late delivery", "Shipping on time"
        ),
        "Late_delivery_risk": late,
        "Category Name": rng.choice(
            ["Sporting Goods", "Apparel", "Footwear", "Outdoors"], n
        ),
        "Order Region": rng.choice(
            ["West of USA", "East of USA", "Central", "South"], n
        ),
        "Shipping Mode": rng.choice(
            ["Standard Class", "First Class", "Second Class", "Same Day"],
            n,
        ),
        "Market": rng.choice(["LATAM", "USCA", "APAC", "Europe"], n),
        "Customer Segment": rng.choice(
            ["Consumer", "Corporate", "Home Office"], n
        ),
        "Department Name": rng.choice(
            ["Fan Shop", "Apparel", "Footwear"], n
        ),
        "Order Item Quantity": rng.integers(1, 6, n),
        "Order Item Discount Rate": rng.uniform(0, .25, n),
        "order date (DateOrders)": pd.date_range(
            "2025-01-01", periods=n, freq="D"
        ).astype(str),
    })


@st.cache_data(show_spinner="Loading and preparing dataset...")
def load_data(
    path: str | None,
    raw_bytes: bytes | None,
) -> tuple[pd.DataFrame, bool, str]:
    """Load, clean and validate data. Return data, demo flag, message."""
    raw = None

    if raw_bytes is not None:
        raw = read_csv_safely(io.BytesIO(raw_bytes))
        source_name = "Uploaded CSV"
    elif path:
        raw = read_csv_safely(path)
        source_name = Path(path).name
    else:
        source_name = "Demo dataset"

    is_demo = raw is None or raw.empty

    if is_demo:
        df = create_demo_data()
        source_message = (
            "Using illustrative demo data because no readable dataset "
            "was found. Upload your DataCo CSV to analyze real records."
        )
    else:
        df = raw.copy()
        source_message = f"Loaded source: {source_name}"

    df.columns = [str(c).strip() for c in df.columns]

    # Remove personal or sensitive fields from the analytics view.
    df = df.drop(
        columns=[c for c in PII_COLUMNS if c in df.columns],
        errors="ignore",
    )

    # Exclude cancelled orders if the status field exists.
    if "Delivery Status" in df.columns:
        status = df["Delivery Status"].astype(str).str.strip().str.lower()
        df = df[~status.eq("shipping canceled")].copy()

    # Numeric conversion.
    numeric_cols = set(MODEL_NUM + [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Late_delivery_risk",
        "Benefit per order",
        "Order Profit Per Order",
        "Sales per customer",
        "Sales",
        "Order Item Quantity",
        "Order Item Discount Rate",
    ])

    for col in numeric_cols.intersection(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive risk from actual versus scheduled transit time if possible.
    actual_col = "Days for shipping (real)"
    scheduled_col = "Days for shipment (scheduled)"

    if actual_col in df.columns and scheduled_col in df.columns:
        df["Delay"] = df[actual_col] - df[scheduled_col]

    if "Late_delivery_risk" not in df.columns:
        if "Delay" in df.columns:
            df["Late_delivery_risk"] = (
                df["Delay"] > 0
            ).astype(int)
        elif "Delivery Status" in df.columns:
            df["Late_delivery_risk"] = (
                df["Delivery Status"].astype(str).str.lower()
                .str.contains("late", na=False)
            ).astype(int)
        else:
            df["Late_delivery_risk"] = 0

    df["Late_delivery_risk"] = pd.to_numeric(
        df["Late_delivery_risk"], errors="coerce"
    )
    df = df.dropna(subset=["Late_delivery_risk"]).copy()
    df["Late_delivery_risk"] = (
        df["Late_delivery_risk"].astype(int).clip(0, 1)
    )

    if "Delay" not in df.columns:
        df["Delay"] = np.nan

    if "Delivery Status" not in df.columns:
        df["Delivery Status"] = np.where(
            df["Late_delivery_risk"].eq(1),
            "Late delivery",
            "On time / early",
        )

    profit_col = next(
        (c for c in PROFIT_CANDIDATES if c in df.columns), None
    )

    if profit_col:
        df["Profit"] = pd.to_numeric(
            df[profit_col], errors="coerce"
        ).fillna(0)
    else:
        df["Profit"] = 0.0

    date_col = "order date (DateOrders)"

    if date_col in df.columns:
        df["order_date_parsed"] = pd.to_datetime(
            df[date_col], errors="coerce"
        )
        df["Order Month"] = (
            df["order_date_parsed"].dt.to_period("M").astype(str)
        )
        df.loc[df["order_date_parsed"].isna(), "Order Month"] = None

    return df.reset_index(drop=True), is_demo, source_message


# ============================================================
# 4. FORMATTING & SHARED HELPERS
# ============================================================

def money(value: float) -> str:
    """Format financial figures in a compact format."""
    if pd.isna(value):
        return "N/A"

    sign = "-" if value < 0 else ""
    value = abs(float(value))

    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{sign}${value / 1_000:.1f}K"

    return f"{sign}${value:,.2f}"


def kpi(
    label: str,
    value: str,
    note: str = "",
    color: str = INK,
) -> str:
    """Render a consistent KPI card."""
    return (
        "<div class='kpi'>"
        f"<div class='label'>{html.escape(label)}</div>"
        f"<div class='value' style='color:{color}'>"
        f"{html.escape(value)}</div>"
        f"<div class='note'>{html.escape(note)}</div>"
        "</div>"
    )


def style_fig(
    fig: go.Figure,
    height: int = 340,
) -> go.Figure:
    """Apply shared chart styling."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=15, r=20, t=25, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, sans-serif",
            color=INK,
            size=12,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
        ),
        hoverlabel=dict(font_size=12),
    )

    fig.update_xaxes(
        gridcolor=LINE,
        zerolinecolor=LINE,
    )
    fig.update_yaxes(
        gridcolor=LINE,
        zerolinecolor=LINE,
    )

    return fig


def rate_table(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Summarize volume, late rate and profit by a dimension."""
    return (
        df.groupby(dimension, dropna=False)
        .agg(
            Orders=("Late_delivery_risk", "size"),
            Late_orders=("Late_delivery_risk", "sum"),
            Late_rate=("Late_delivery_risk", "mean"),
            Profit=("Profit", "sum"),
        )
        .reset_index()
    )


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "Late_rate" in out.columns:
        out["Late_rate"] = (out["Late_rate"] * 100).round(1)
        out = out.rename(columns={"Late_rate": "Late rate (%)"})

    out = out.rename(columns={"Late_orders": "Late orders"})

    if "Profit" in out.columns:
        out["Profit"] = out["Profit"].round(2)

    return out


def min_volume(df: pd.DataFrame) -> int:
    return max(10, int(len(df) * 0.01))


# ============================================================
# 5. ANALYTICS AGENT
# ============================================================

from dataclasses import dataclass, field


@dataclass
class AgentResult:
    title: str
    finding: str
    color: str = BRAND
    table: pd.DataFrame | None = None
    fig: go.Figure | None = None
    actions: list[str] = field(default_factory=list)


def summary_tool(df: pd.DataFrame) -> AgentResult:
    total = len(df)
    late_count = int(df["Late_delivery_risk"].sum())
    late_rate = late_count / total if total else 0

    findings = [
        f"**{late_count:,} of {total:,} orders "
        f"({late_rate:.1%})** are flagged as late."
    ]

    actions = [
        "Review the highest-risk shipping modes and investigate "
        "the largest contributing regions.",
        "Monitor delivery performance regularly and compare "
        "late rates with operational targets.",
    ]

    if "Shipping Mode" in df.columns:
        rates = rate_table(df, "Shipping Mode")
        eligible = rates[rates["Orders"] >= min_volume(df)]
        if not eligible.empty:
            worst = eligible.sort_values(
                "Late_rate", ascending=False
            ).iloc[0]
            findings.append(
                f"**{worst['Shipping Mode']}** has the highest "
                f"late rate among sufficiently represented modes "
                f"({worst['Late_rate']:.1%})."
            )

    late_df = df[df["Late_delivery_risk"] == 1]

    if "Order Region" in df.columns and not late_df.empty:
        region = late_df["Order Region"].value_counts()
        if not region.empty:
            findings.append(
                f"**{region.index[0]}** contributes the most late "
                f"orders ({int(region.iloc[0]):,})."
            )

    findings.append(
        f"Recorded profit associated with late orders: "
        f"**{money(late_df['Profit'].sum())}**. This is profit "
        "associated with those orders, not an estimate of losses "
        "caused by delivery delays."
    )

    return AgentResult(
        "Executive Operations Briefing",
        " ".join(findings),
        BRAND,
        actions=actions,
    )


def dimension_tool(
    df: pd.DataFrame,
    dimension: str,
    label: str,
    lowest: bool = False,
) -> AgentResult:
    table = rate_table(df, dimension)

    eligible = table[table["Orders"] >= min_volume(df)]
    if eligible.empty:
        eligible = table

    if eligible.empty:
        return AgentResult(
            f"Risk by {label}",
            "There is insufficient data for this comparison.",
            WARN,
        )

    ordered = eligible.sort_values(
        "Late_rate", ascending=lowest
    )
    picked = ordered.iloc[0]

    volume_rank = table.sort_values(
        "Late_orders", ascending=False
    ).iloc[0]

    direction = "lowest" if lowest else "highest"

    finding = (
        f"**{picked[dimension]}** has the {direction} late-delivery "
        f"rate by {label}: **{picked['Late_rate']:.1%}** across "
        f"{int(picked['Orders']):,} orders. "
        f"**{volume_rank[dimension]}** contributes the most late "
        f"orders by volume ({int(volume_rank['Late_orders']):,})."
    )

    shown = ordered.head(10).copy()
    shown[dimension] = shown[dimension].astype(str)

    fig = px.bar(
        shown,
        x="Late_rate",
        y=dimension,
        orientation="h",
        text=shown["Late_rate"].map("{:.1%}".format),
        labels={
            "Late_rate": "Late-delivery rate",
            dimension: label.title(),
        },
    )

    fig.update_traces(
        marker_color=[
            RISK if i == 0 else BRAND
            for i in range(len(shown))
        ],
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%", range=[0, 1.05])

    actions = [
        f"Investigate operating conditions and carrier performance "
        f"for {picked[dimension]}. Compare order volume and promised "
        f"delivery time before changing service policies."
    ]

    return AgentResult(
        f"Delivery Risk by {label.title()}",
        finding,
        RISK,
        display_table(shown),
        fig,
        actions,
    )


def root_cause_tool(df: pd.DataFrame) -> AgentResult:
    required = [
        "Shipping Mode",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
    ]

    if not all(c in df.columns for c in required):
        return AgentResult(
            "Delivery Delay Root-Cause Analysis",
            "Actual transit-time or scheduled delivery columns are "
            "missing. Upload a dataset with both fields to compare "
            "promised and actual delivery duration.",
            WARN,
        )

    late = df[df["Late_delivery_risk"] == 1]
    delay_values = pd.to_numeric(
        late["Delay"], errors="coerce"
    ).dropna()

    if late.empty or delay_values.empty:
        return AgentResult(
            "Delivery Delay Root-Cause Analysis",
            "There are not enough late orders with valid delay "
            "values in the current filtered view.",
            WARN,
        )

    grouped = df.groupby("Shipping Mode", dropna=False).agg(
        Orders=("Late_delivery_risk", "size"),
        Late_rate=("Late_delivery_risk", "mean"),
        Promised_days=("Days for shipment (scheduled)", "mean"),
        Actual_days=("Days for shipping (real)", "mean"),
    ).reset_index()

    grouped = grouped.sort_values("Late_rate", ascending=False)
    worst = grouped.iloc[0]

    finding = (
        f"Late orders exceed the scheduled duration by an average "
        f"of **{delay_values.mean():.1f} days**. The highest observed "
        f"late rate is for **{worst['Shipping Mode']}** "
        f"({worst['Late_rate']:.1%}). This comparison identifies an "
        "association, not a proven causal explanation."
    )

    fig = go.Figure()
    fig.add_bar(
        name="Scheduled",
        x=grouped["Shipping Mode"],
        y=grouped["Promised_days"],
        marker_color=BRAND,
    )
    fig.add_bar(
        name="Actual",
        x=grouped["Shipping Mode"],
        y=grouped["Actual_days"],
        marker_color=RISK,
    )

    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Average transit days")

    actions = [
        "Compare carrier transit times against service commitments.",
        "Investigate warehouse processing, carrier handoffs, "
        "regional congestion and demand peaks before assigning cause.",
    ]

    return AgentResult(
        "Delivery Delay Root-Cause Analysis",
        finding,
        RISK,
        display_table(grouped),
        fig,
        actions,
    )


def financial_tool(df: pd.DataFrame) -> AgentResult:
    late = df[df["Late_delivery_risk"] == 1]
    on_time = df[df["Late_delivery_risk"] == 0]

    if late.empty:
        return AgentResult(
            "Financial Impact of Late Deliveries",
            "No late orders are present in the current view.",
            OK,
        )

    loss_orders = late[late["Profit"] < 0]
    average_late = late["Profit"].mean()
    average_on_time = on_time["Profit"].mean() if not on_time.empty else np.nan

    finding = (
        f"Late orders have recorded aggregate profit of "
        f"**{money(late['Profit'].sum())}**. "
        f"**{len(loss_orders):,}** late orders have negative recorded "
        f"profit, with combined profit of **{money(loss_orders['Profit'].sum())}**. "
        f"Average recorded profit per order is **{money(average_late)}** "
        f"for late orders versus **{money(average_on_time)}** for "
        "on-time orders. These values do not establish that delays "
        "caused the difference."
    )

    fig = go.Figure()
    fig.add_bar(
        x=["Late orders", "On-time orders"],
        y=[
            average_late,
            average_on_time if not pd.isna(average_on_time) else 0,
        ],
        marker_color=[RISK, OK],
        text=[
            money(average_late),
            money(average_on_time),
        ],
        textposition="outside",
    )
    fig.update_yaxes(title="Average recorded profit per order")

    return AgentResult(
        "Financial Impact of Late Deliveries",
        finding,
        WARN,
        fig=fig,
        actions=[
            "Review negative-profit late orders alongside shipping "
            "costs, discounts, order value and operational service levels."
        ],
    )


def trend_tool(df: pd.DataFrame) -> AgentResult:
    if (
        "order_date_parsed" not in df.columns
        or df["order_date_parsed"].notna().sum() == 0
    ):
        return AgentResult(
            "Delivery Performance Trend",
            "No valid order-date field is available for a time-series "
            "analysis.",
            WARN,
        )

    dated = df.dropna(subset=["order_date_parsed"]).copy()
    dated["Period"] = (
        dated["order_date_parsed"].dt.to_period("M").astype(str)
    )

    monthly = dated.groupby("Period").agg(
        Orders=("Late_delivery_risk", "size"),
        Late_rate=("Late_delivery_risk", "mean"),
    ).reset_index().sort_values("Period")

    if len(monthly) < 2:
        return AgentResult(
            "Delivery Performance Trend",
            "At least two distinct months are required for a meaningful "
            "month-to-month comparison.",
            WARN,
        )

    fig = px.line(
        monthly,
        x="Period",
        y="Late_rate",
        markers=True,
        labels={
            "Period": "Order month",
            "Late_rate": "Late-delivery rate",
        },
    )
    fig.update_traces(line_color=BRAND)
    fig.update_yaxes(tickformat=".0%")

    return AgentResult(
        "Monthly Delivery Performance",
        f"Delivery risk is tracked across {len(monthly)} months. "
        "Review changes in order volume and mix alongside the late "
        "rate before attributing any change to a specific cause.",
        BRAND,
        display_table(monthly),
        fig,
        ["Review capacity and carrier performance before seasonal peaks."],
    )


def resolve_dimensions(question: str, columns: set[str]):
    dimensions = []

    if "region" in question:
        dimensions.append(("Order Region", "region"))
    if "market" in question:
        dimensions.append(("Market", "market"))
    if "categor" in question or "product" in question:
        dimensions.append(("Category Name", "category"))
    if "shipping" in question or "mode" in question or "carrier" in question:
        dimensions.append(("Shipping Mode", "shipping mode"))
    if "segment" in question:
        dimensions.append(("Customer Segment", "customer segment"))

    return [
        (column, label)
        for column, label in dimensions
        if column in columns
    ]


def run_agent(question: str, df: pd.DataFrame):
    """Route questions to local, data-backed analysis functions."""
    q = question.lower().strip()
    results = []
    steps = []

    lowest = any(
        word in q for word in ["lowest", "least", "best", "safest"]
    )

    dimensions = resolve_dimensions(q, set(df.columns))

    for column, label in dimensions:
        steps.append(f"Compared late-delivery rates by {label}.")
        results.append(dimension_tool(df, column, label, lowest))

    if any(word in q for word in [
        "why", "reason", "cause", "delay", "delayed", "root",
    ]):
        steps.append("Compared scheduled and actual delivery duration.")
        results.append(root_cause_tool(df))

    if any(word in q for word in [
        "profit", "loss", "money", "financial", "revenue", "cost",
    ]):
        steps.append("Calculated recorded profit exposure.")
        results.append(financial_tool(df))

    if any(word in q for word in ["trend", "month", "monthly", "time"]):
        steps.append("Analyzed delivery performance over time.")
        results.append(trend_tool(df))

    if any(word in q for word in ["summary", "overview", "overall", "briefing"]):
        steps.append("Prepared the executive operations summary.")
        results.append(summary_tool(df))

    if not results:
        steps = [
            "Reviewed overall delivery performance.",
            "Checked delay and financial indicators.",
        ]
        results = [
            summary_tool(df),
            root_cause_tool(df),
            financial_tool(df),
        ]

    return steps, results


def render_result(result: AgentResult, key: str):
    st.markdown(f"#### {result.title}")

    st.markdown(
        "<div class='finding' style='--c:"
        + result.color
        + "'>"
        + result.finding
        + "</div>",
        unsafe_allow_html=True,
    )

    if result.fig is not None:
        st.plotly_chart(
            style_fig(result.fig, 320),
            use_container_width=True,
            key=f"result-{key}",
        )

    if result.actions:
        st.markdown("**Recommended next steps**")
        for action in result.actions:
            st.markdown(f"- {action}")

    if result.table is not None:
        with st.expander("View supporting data"):
            st.dataframe(
                result.table,
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# 6. LOAD DATA AND SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🚚 Control Tower")
    st.caption("Supply Chain Intelligence Platform")
    st.divider()

    st.markdown("### Data Source")

    uploaded = st.file_uploader(
        "Upload supply chain CSV",
        type=["csv"],
        help="Upload DataCoSupplyChainDataset.csv or a compatible file.",
    )

    st.markdown(f"[GitHub Repository]({GITHUB_URL})")
    st.divider()

default_path = find_default_file()

full, is_demo, source_message = load_data(
    default_path,
    uploaded.getvalue() if uploaded is not None else None,
)

if full.empty:
    st.error("The dataset contains no usable orders after cleaning.")
    st.stop()

with st.sidebar:
    st.markdown("### Dashboard Filters")
    st.caption("Leave selections empty to include all values.")

    view = full.copy()

    filter_fields = [
        ("Market", "Market"),
        ("Region", "Order Region"),
        ("Shipping Mode", "Shipping Mode"),
        ("Customer Segment", "Customer Segment"),
    ]

    for label, column in filter_fields:
        if column in full.columns:
            options = sorted(
                full[column].dropna().astype(str).unique().tolist()
            )

            selected = st.multiselect(
                label,
                options,
                key=f"filter-{column}",
                placeholder="All values",
            )

            if selected:
                view = view[
                    view[column].astype(str).isin(selected)
                ]

    st.divider()
    st.caption("Privacy: common customer identifiers are removed.")


if view.empty:
    st.warning(
        "No orders match your current filters. Change the sidebar "
        "selections to continue."
    )
    st.stop()


# ============================================================
# 7. MAIN HEADER AND KPIs
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">
            Supply Chain Intelligence Platform
        </div>
        <div class="hero-title">
            Supply Chain Control Tower
        </div>
        <div class="hero-subtitle">
            Monitor delivery performance, investigate operational
            bottlenecks, measure financial exposure and evaluate
            late-delivery risk through one unified analytics workspace.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="section-heading">Executive Overview</div>
    <div class="section-caption">
        Key performance indicators for the current filtered dataset.
        Currently analyzing <b>{len(view):,} orders</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

if is_demo:
    st.info(
        "Demo mode: these are illustrative records, not real operational "
        "results. Upload your dataset to analyze actual orders."
    )
else:
    st.success(source_message)

n = len(view)
late_df = view[view["Late_delivery_risk"] == 1]
n_late = len(late_df)
n_not_late = n - n_late

late_rate = n_late / n if n else 0
not_late_rate = n_not_late / n if n else 0

avg_late_days = pd.to_numeric(
    late_df["Delay"], errors="coerce"
).mean()

avg_late_str = (
    f"{avg_late_days:.1f} days"
    if not pd.isna(avg_late_days)
    else "N/A"
)

k1, k2, k3, k4, k5 = st.columns(5)

k1.markdown(
    kpi("Total Orders", f"{n:,}", "Orders in current view"),
    unsafe_allow_html=True,
)
k2.markdown(
    kpi(
        "Late-Delivery Rate",
        f"{late_rate:.1%}",
        f"{n_late:,} late orders",
        RISK,
    ),
    unsafe_allow_html=True,
)
k3.markdown(
    kpi(
        "Not Flagged as Late",
        f"{n_not_late:,}",
        f"{not_late_rate:.1%} of current orders",
        OK,
    ),
    unsafe_allow_html=True,
)
k4.markdown(
    kpi("Average Days Late", avg_late_str, "Late orders only", WARN),
    unsafe_allow_html=True,
)
k5.markdown(
    kpi(
        "Recorded Profit: Late Orders",
        money(late_df["Profit"].sum()),
        f"{int((late_df['Profit'] < 0).sum()):,} negative-profit orders",
        WARN,
    ),
    unsafe_allow_html=True,
)

st.write("")

# ============================================================
# 8. TABS
# ============================================================

tab_agent, tab_overview, tab_model = st.tabs([
    "🤖 AI Operations Agent",
    "📊 Performance Analytics",
    "🎯 Predictive Risk Engine",
])


# ============================================================
# 9. AI OPERATIONS AGENT
# ============================================================

with tab_agent:
    st.markdown(
        """
        <div class="section-heading">
            AI Operations & Decision Support
        </div>
        <div class="section-caption">
            Automated briefings, delivery-risk comparisons, delay analysis
            and financial summaries. The local agent uses Python rules and
            calculations; it is not an external generative AI model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="briefing-head">
            <h3>Executive Operations Briefing</h3>
            <p>Summary of the current filtered orders.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for index, result in enumerate([
        summary_tool(view),
        root_cause_tool(view),
        financial_tool(view),
    ]):
        render_result(result, f"brief-{index}")

    st.divider()

    st.markdown("### Ask the Operations Agent")
    st.caption(
        "Ask about delivery delays, regions, shipping modes, profit or trends."
    )

    examples = [
        "Which region has the highest risk?",
        "Which shipping mode is worst?",
        "Why are orders delayed?",
        "Show financial exposure",
        "Show monthly trend",
    ]

    example_cols = st.columns(5)

    for col, example in zip(example_cols, examples):
        if col.button(
            example,
            key=f"example-{example}",
            use_container_width=True,
        ):
            st.session_state["agent_question"] = example

    question = st.text_input(
        "Enter your question",
        key="agent_question",
        placeholder="e.g. Which market has the highest late-delivery rate?",
    )

    if question.strip():
        steps, results = run_agent(question, view)

        with st.expander("Analysis Workflow"):
            for index, step in enumerate(steps, 1):
                st.write(f"{index}. {step}")

        for index, result in enumerate(results):
            render_result(result, f"question-{index}")


# ============================================================
# 10. PERFORMANCE ANALYTICS
# ============================================================

with tab_overview:
    st.markdown(
        """
        <div class="section-heading">
            Delivery Performance Analytics
        </div>
        <div class="section-caption">
            Explore delivery-risk patterns, transit-time distributions,
            monthly trends and individual order records.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Delivery Risk by Shipping Mode")

        if "Shipping Mode" in view.columns:
            table = rate_table(view, "Shipping Mode").sort_values(
                "Late_rate", ascending=False
            )

            fig = px.bar(
                table,
                x="Shipping Mode",
                y="Late_rate",
                text=table["Late_rate"].map("{:.1%}".format),
                labels={
                    "Late_rate": "Late-delivery rate",
                    "Shipping Mode": "Shipping mode",
                },
            )

            fig.update_traces(
                marker_color=[
                    RISK if i == 0 else BRAND
                    for i in range(len(table))
                ],
                textposition="outside",
                cliponaxis=False,
            )
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05])

            st.plotly_chart(
                style_fig(fig),
                use_container_width=True,
                key="overview-shipping",
            )
        else:
            st.info("Shipping Mode is not available in this dataset.")

    with right:
        st.markdown("#### Distribution of Delivery Delays")

        if view["Delay"].notna().any():
            plot_data = view.copy()
            plot_data["Delay"] = pd.to_numeric(
                plot_data["Delay"], errors="coerce"
            )
            plot_data = plot_data.dropna(subset=["Delay"])

            fig = px.histogram(
                plot_data,
                x="Delay",
                color="Delivery Status",
                nbins=20,
                barmode="overlay",
                opacity=.8,
                labels={
                    "Delay": "Actual minus scheduled days",
                    "count": "Orders",
                },
            )

            fig.update_layout(bargap=.08)

            st.plotly_chart(
                style_fig(fig),
                use_container_width=True,
                key="overview-delay",
            )
        else:
            st.info(
                "Actual and scheduled transit times are required "
                "to plot delivery delays."
            )

    st.divider()

    st.markdown("#### Monthly Delivery Performance")

    if (
        "order_date_parsed" in view.columns
        and view["order_date_parsed"].notna().any()
    ):
        dated = view.dropna(subset=["order_date_parsed"]).copy()
        dated["Period"] = (
            dated["order_date_parsed"].dt.to_period("M").astype(str)
        )

        monthly = dated.groupby("Period").agg(
            Orders=("Late_delivery_risk", "size"),
            Late_rate=("Late_delivery_risk", "mean"),
        ).reset_index().sort_values("Period")

        if len(monthly) >= 2:
            fig = px.line(
                monthly,
                x="Period",
                y="Late_rate",
                markers=True,
                labels={
                    "Period": "Order month",
                    "Late_rate": "Late-delivery rate",
                },
            )
            fig.update_traces(line_color=BRAND)
            fig.update_yaxes(tickformat=".0%")

            st.plotly_chart(
                style_fig(fig, 350),
                use_container_width=True,
                key="overview-monthly",
            )
        else:
            st.info("At least two months are required for a trend.")
    else:
        st.info("No valid order dates are available for trend analysis.")

    st.divider()

    st.markdown("#### High-Risk Orders")

    high_risk = view[view["Late_delivery_risk"] == 1].copy()

    st.caption(
        f"{len(high_risk):,} orders are currently flagged as late."
    )

    st.download_button(
        "⬇ Download Late Orders CSV",
        data=high_risk.to_csv(index=False).encode("utf-8"),
        file_name="late_delivery_orders.csv",
        mime="text/csv",
        key="download-late-orders",
    )

    st.dataframe(
        high_risk.head(100),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Explore all filtered orders"):
        st.caption(
            "The table below displays the first 500 rows of the current view."
        )
        st.dataframe(
            view.head(500),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download All Filtered Orders CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="filtered_supply_chain_orders.csv",
            mime="text/csv",
            key="download-all-orders",
        )


# ============================================================
# 11. PREDICTIVE RISK ENGINE
# ============================================================

@st.cache_resource(show_spinner="Training the risk prediction model...")
def train_model(
    data: pd.DataFrame,
    data_key: str,
) -> dict:
    """Train on available order-time fields and evaluate held-out data."""
    del data_key  # Included in the cache key to distinguish data sources.

    cat = [c for c in MODEL_CAT if c in data.columns]
    num = [c for c in MODEL_NUM if c in data.columns]
    features = cat + num

    if not features:
        return {"error": "No supported order-time features are available."}

    working = data[features + ["Late_delivery_risk"]].copy()
    y = working["Late_delivery_risk"].astype(int)

    if y.nunique() < 2:
        return {
            "error": (
                "The target contains only one class. The model needs "
                "both late and non-late examples."
            )
        }

    if y.value_counts().min() < 2 or len(working) < 10:
        return {
            "error": (
                "There are too few examples in one or more classes "
                "for a reliable train/test split."
            )
        }

    X = working[features].copy()

    for col in cat:
        X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)

    for col in num:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        if X[col].notna().sum() == 0:
            X[col] = X[col].fillna(0)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=.20,
            random_state=42,
            stratify=y,
        )

        transformers = []

        if cat:
            categorical_pipeline = Pipeline([
                (
                    "encode",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ])

            transformers.append(
                ("cat", categorical_pipeline, cat)
            )

        if num:
            numeric_pipeline = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
            ])

            transformers.append(
                ("num", numeric_pipeline, num)
            )

        preprocessor = ColumnTransformer(
            transformers,
            remainder="drop",
        )

        # The encoded category fields are treated as categorical
        # by the HistGradientBoosting model.
        model = Pipeline([
            ("pre", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    categorical_features=(
                        [True] * len(cat) + [False] * len(num)
                    ),
                    max_iter=120,
                    learning_rate=.08,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ])

        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= .5).astype(int)

        auc = (
            roc_auc_score(y_test, probabilities)
            if y_test.nunique() == 2
            else np.nan
        )

        defaults = {}

        for col in cat:
            modes = X[col].mode()
            defaults[col] = (
                str(modes.iloc[0]) if not modes.empty else "__MISSING__"
            )

        for col in num:
            median = X[col].median()
            defaults[col] = float(median) if not pd.isna(median) else 0.0

        return {
            "model": model,
            "cat": cat,
            "num": num,
            "features": features,
            "defaults": defaults,
            "accuracy": accuracy_score(y_test, predictions),
            "auc": auc,
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "baseline": max(
                y_test.mean(), 1 - y_test.mean()
            ),
            "n_test": len(y_test),
            "error": None,
        }

    except (ValueError, TypeError, KeyError) as exc:
        return {
            "error": (
                "Model training could not be completed for this dataset. "
                f"Details: {str(exc)[:250]}"
            )
        }


with tab_model:
    st.markdown(
        """
        <div class="section-heading">
            Late-Delivery Risk Prediction
        </div>
        <div class="section-caption">
            Train a model using supported order-time variables, review
            held-out evaluation metrics and simulate a new order.
            Actual delivery outcomes are not used as prediction features.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "Model scores are evaluation results, not guarantees for future "
        "orders. Demo-data scores are illustrative only."
    )

    # Build a cache key that changes when the loaded dataset changes.
    if uploaded is not None:
        model_data_key = f"upload-{uploaded.name}-{len(full)}"
    else:
        model_data_key = f"local-{default_path}-{len(full)}"

    model_result = train_model(full, model_data_key)

    if model_result.get("error"):
        st.error(model_result["error"])
        st.info(
            "Check that the dataset has both target classes and at least "
            "one supported order-time feature."
        )
    else:
        metrics = st.columns(4)

        metrics[0].markdown(
            kpi(
                "Test Accuracy",
                f"{model_result['accuracy']:.1%}",
                f"Baseline: {model_result['baseline']:.1%}",
            ),
            unsafe_allow_html=True,
        )

        auc = model_result["auc"]
        metrics[1].markdown(
            kpi(
                "ROC-AUC",
                f"{auc:.3f}" if not pd.isna(auc) else "N/A",
                "Held-out discrimination score",
                BRAND,
            ),
            unsafe_allow_html=True,
        )

        metrics[2].markdown(
            kpi(
                "Late-Class Recall",
                f"{model_result['recall']:.1%}",
                "At a 0.50 probability threshold",
                WARN,
            ),
            unsafe_allow_html=True,
        )

        metrics[3].markdown(
            kpi(
                "Test Set",
                f"{model_result['n_test']:,}",
                "Held-out orders (20%)",
            ),
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown("### New Order Risk Simulation")
        st.caption(
            "Change the order attributes and calculate the model's "
            "estimated probability of late delivery."
        )

        with st.form("risk_simulation_form"):
            left, right = st.columns(2)
            inputs = {}

            with left:
                for col in model_result["cat"]:
                    raw_options = (
                        full[col].dropna().astype(str).unique().tolist()
                    )
                    options = sorted(raw_options)

                    if not options:
                        options = [
                            str(model_result["defaults"][col])
                        ]

                    default = str(model_result["defaults"][col])
                    default_index = (
                        options.index(default) if default in options else 0
                    )

                    inputs[col] = st.selectbox(
                        col,
                        options,
                        index=default_index,
                    )

            with right:
                for col in model_result["num"]:
                    default = float(model_result["defaults"][col])

                    inputs[col] = st.number_input(
                        col,
                        value=default,
                        step=1.0,
                    )

            submitted = st.form_submit_button(
                "Calculate Late-Delivery Risk",
                use_container_width=True,
            )

        if submitted:
            simulation = pd.DataFrame([inputs])

            for col in model_result["cat"]:
                simulation[col] = (
                    simulation[col].astype("string")
                    .fillna("__MISSING__").astype(str)
                )

            for col in model_result["num"]:
                simulation[col] = pd.to_numeric(
                    simulation[col], errors="coerce"
                )

            try:
                probability = float(
                    model_result["model"].predict_proba(simulation)[0, 1]
                )

                st.markdown("#### Prediction Result")

                result_col, chart_col = st.columns([1, 1])

                with result_col:
                    st.metric(
                        "Estimated Late-Delivery Probability",
                        f"{probability:.1%}",
                    )

                    if probability >= .5:
                        st.error(
                            "Higher predicted risk at the 50% threshold. "
                            "Review delivery commitments and available "
                            "expediting options."
                        )
                    else:
                        st.success(
                            "Lower predicted risk at the 50% threshold. "
                            "Continue monitoring the order against "
                            "operational service targets."
                        )

                with chart_col:
                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            number={"suffix": "%"},
                            title={"text": "Late-Delivery Probability"},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": BRAND},
                                "steps": [
                                    {
                                        "range": [0, 50],
                                        "color": "#DDF5EC",
                                    },
                                    {
                                        "range": [50, 100],
                                        "color": "#FCE5E7",
                                    },
                                ],
                                "threshold": {
                                    "line": {"color": RISK, "width": 3},
                                    "value": 50,
                                },
                            },
                        )
                    )

                    st.plotly_chart(
                        style_fig(gauge, 260),
                        use_container_width=True,
                        key="risk-gauge",
                    )

                st.caption(
                    "The 50% threshold is a default decision rule, not "
                    "a validated business cutoff. Validate the model "
                    "before using its predictions for operational decisions."
                )

            except (ValueError, TypeError, KeyError) as exc:
                st.error(
                    "The order could not be scored. Check the selected "
                    f"values and model features. Details: {str(exc)[:200]}"
                )


# ============================================================
# 12. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:8px 0 18px 0;">
        <div style="font-weight:700; color:#172B4D;">
            Supply Chain Control Tower
        </div>
        <div style="font-size:12px; color:#64748B; margin-top:6px;">
            Delivery Analytics · Operational Intelligence · Risk Prediction
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
