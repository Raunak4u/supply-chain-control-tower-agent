
from __future__ import annotations

import html
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

GITHUB_URL = (
    "https://github.com/Raunak4u/"
    "supply-chain-control-tower-agent"
)

HF_URL = (
    "https://huggingface.co/spaces/"
    "RAUNAK-KUMAR/supply-chain-control-tower-agent"
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

INK = "#0F1B2D"
MUTED = "#5B6B80"
LINE = "#E2E8F0"
BRAND = "#0E6BA8"
RISK = "#D1495B"
OK = "#2A9D8F"
WARN = "#E9A23B"

st.set_page_config(
    page_title="Supply Chain Control Tower",
    page_icon="🚚",
    layout="wide",
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap'
    );

    html, body, .stApp, .stMarkdown, button, input, label {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}

    .stApp {{
        background: #F3F6FA;
        color: {INK};
    }}

    [data-testid="stSidebar"] {{
        background: #FFFFFF;
        border-right: 1px solid {LINE};
    }}

    .block-container {{
        padding-top: 1.8rem;
        max-width: 1320px;
    }}

    .page-title {{
        font-size: 32px;
        font-weight: 800;
        color: {INK};
        margin-bottom: 0;
    }}

    .page-sub {{
        color: {MUTED};
        margin-top: 4px;
        margin-bottom: 20px;
    }}

    .kpi {{
        background: white;
        border: 1px solid {LINE};
        border-radius: 12px;
        padding: 15px;
        min-height: 105px;
    }}

    .kpi .label {{
        color: {MUTED};
        font-size: 12px;
        font-weight: 600;
    }}

    .kpi .value {{
        font-size: 25px;
        font-weight: 800;
        overflow-wrap: anywhere;
    }}

    .kpi .note {{
        font-size: 12px;
        color: {MUTED};
    }}

    .finding {{
        background: white;
        border: 1px solid {LINE};
        border-left: 4px solid var(--c, {BRAND});
        border-radius: 10px;
        padding: 14px;
        line-height: 1.6;
        margin-bottom: 12px;
    }}

    .briefing-head {{
        border-left: 4px solid {BRAND};
        padding-left: 12px;
        margin-bottom: 15px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATA LOADING
# ============================================================

def find_default_file() -> Optional[str]:
    """Find a supported dataset in common project folders."""

    base = Path(__file__).resolve().parent

    folders = [
        base,
        base / "data",
        Path.cwd(),
        Path.cwd() / "data",
    ]

    for folder in folders:
        for filename in FILE_CANDIDATES:
            path = folder / filename

            if path.exists():
                return str(path)

    return None


def read_csv_source(source) -> pd.DataFrame:
    """Read a CSV using supported text encodings."""

    for encoding in ["utf-8", "latin-1"]:
        try:
            if hasattr(source, "seek"):
                source.seek(0)

            return pd.read_csv(
                source,
                encoding=encoding,
                low_memory=False,
            )

        except UnicodeDecodeError:
            continue

    raise ValueError("Unable to decode the uploaded CSV.")


@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_data(
    path: Optional[str],
    raw: Optional[bytes],
) -> pd.DataFrame:

    if raw is not None:
        df = read_csv_source(io.BytesIO(raw))

    elif path is not None:
        df = read_csv_source(path)

    else:
        raise ValueError("No dataset was supplied.")

    # Clean column names.
    df.columns = [str(c).strip() for c in df.columns]

    # Remove unnecessary personal information.
    df = df.drop(
        columns=[
            c for c in PII_COLUMNS
            if c in df.columns
        ],
        errors="ignore",
    )

    actual_col = "Days for shipping (real)"
    scheduled_col = "Days for shipment (scheduled)"

    # Build the target if it does not already exist.
    if "Late_delivery_risk" not in df.columns:

        if (
            actual_col in df.columns
            and scheduled_col in df.columns
        ):
            actual = pd.to_numeric(
                df[actual_col], errors="coerce"
            )

            scheduled = pd.to_numeric(
                df[scheduled_col], errors="coerce"
            )

            df["Late_delivery_risk"] = (
                actual > scheduled
            ).astype(int)

        else:
            raise ValueError(
                "Dataset needs Late_delivery_risk or both "
                "actual and scheduled shipping-day columns."
            )

    # Validate the target.
    df["Late_delivery_risk"] = pd.to_numeric(
        df["Late_delivery_risk"],
        errors="coerce",
    )

    df = df[
        df["Late_delivery_risk"].isin([0, 1])
    ].copy()

    df["Late_delivery_risk"] = (
        df["Late_delivery_risk"].astype(int)
    )

    # Remove cancelled orders.
    if "Delivery Status" in df.columns:

        cancelled = (
            df["Delivery Status"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq("shipping canceled")
        )

        df = df[~cancelled].copy()

    else:

        df["Delivery Status"] = np.where(
            df["Late_delivery_risk"].eq(1),
            "Late delivery",
            "On time",
        )

    # Calculate delay in days.
    if (
        actual_col in df.columns
        and scheduled_col in df.columns
    ):
        df["Delay"] = (
            pd.to_numeric(
                df[actual_col], errors="coerce"
            )
            -
            pd.to_numeric(
                df[scheduled_col], errors="coerce"
            )
        )

    else:
        df["Delay"] = np.nan

    # Profit.
    profit_col = next(
        (
            col for col in PROFIT_CANDIDATES
            if col in df.columns
        ),
        None,
    )

    if profit_col:
        df["Profit"] = pd.to_numeric(
            df[profit_col], errors="coerce"
        ).fillna(0)

    else:
        df["Profit"] = 0.0

    # Parse order dates.
    date_col = "order date (DateOrders)"

    if date_col in df.columns:

        df["order_date_parsed"] = pd.to_datetime(
            df[date_col],
            errors="coerce",
        )

        df["Order Month"] = (
            df["order_date_parsed"]
            .dt.to_period("M")
            .astype(str)
        )

        df.loc[
            df["order_date_parsed"].isna(),
            "Order Month",
        ] = np.nan

    return df.reset_index(drop=True)


# ============================================================
# 4. MODEL TRAINING
# ============================================================

@st.cache_resource(show_spinner="Training risk prediction model...")
def train_model(
    _df: pd.DataFrame,
    data_key: str,
) -> dict:

    categorical = [
        c for c in MODEL_CAT
        if c in _df.columns
    ]

    numerical = [
        c for c in MODEL_NUM
        if c in _df.columns
    ]

    feature_cols = categorical + numerical

    if not feature_cols:
        raise ValueError(
            "No supported model features were found."
        )

    X = _df[feature_cols].copy()

    y = pd.to_numeric(
        _df["Late_delivery_risk"],
        errors="coerce",
    )

    valid = y.isin([0, 1])

    X = X.loc[valid].copy()
    y = y.loc[valid].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "The model needs both late and on-time orders."
        )

    if len(y) < 20:
        raise ValueError(
            "At least 20 usable rows are needed."
        )

    # Prepare feature types.
    for col in categorical:
        X[col] = (
            X[col]
            .astype("string")
            .fillna("Unknown")
            .astype(str)
        )

    for col in numerical:
        X[col] = pd.to_numeric(
            X[col], errors="coerce"
        )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    transformers = []

    if categorical:

        cat_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        transformers.append(
            ("categorical", cat_pipeline, categorical)
        )

    if numerical:

        num_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
            ]
        )

        transformers.append(
            ("numerical", num_pipeline, numerical)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    try:
        auc = roc_auc_score(
            y_test,
            probabilities,
        )
    except ValueError:
        auc = np.nan

    defaults = {}

    for col in categorical:
        modes = X[col].mode()

        defaults[col] = (
            modes.iloc[0]
            if not modes.empty
            else "Unknown"
        )

    for col in numerical:
        median = X[col].median()

        defaults[col] = (
            float(median)
            if pd.notna(median)
            else 0.0
        )

    return {
        "model": model,
        "cat": categorical,
        "num": numerical,
        "feature_cols": feature_cols,
        "defaults": defaults,
        "accuracy": accuracy_score(
            y_test, predictions
        ),
        "auc": auc,
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "baseline": max(
            y_test.mean(),
            1 - y_test.mean(),
        ),
        "n_test": len(y_test),
    }


# ============================================================
# 5. REUSABLE HELPERS
# ============================================================

def money(value) -> str:

    if pd.isna(value):
        return "n/a"

    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)

    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"{sign}${value / 1_000:.1f}K"

    return f"{sign}${value:,.0f}"


def kpi(
    label: str,
    value: str,
    note: str = "",
    color: str = INK,
) -> str:

    return (
        f"<div class='kpi'>"
        f"<div class='label'>{html.escape(str(label))}</div>"
        f"<div class='value' style='color:{color}'>"
        f"{html.escape(str(value))}</div>"
        f"<div class='note'>{html.escape(str(note))}</div>"
        f"</div>"
    )


def safe_bold_html(text: str) -> str:
    escaped = html.escape(str(text))

    return re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        escaped,
    )


def style_fig(
    fig: go.Figure,
    height: int = 330,
) -> go.Figure:

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Plus Jakarta Sans, sans-serif",
            color=INK,
            size=12,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
        ),
    )

    fig.update_xaxes(gridcolor=LINE)
    fig.update_yaxes(gridcolor=LINE)

    return fig


def rate_table(
    data: pd.DataFrame,
    dimension: str,
) -> pd.DataFrame:

    return (
        data.groupby(
            dimension,
            dropna=False,
        )
        .agg(
            Orders=("Late_delivery_risk", "size"),
            Late_orders=("Late_delivery_risk", "sum"),
            Late_rate=("Late_delivery_risk", "mean"),
            Profit=("Profit", "sum"),
        )
        .reset_index()
    )


def min_volume(data: pd.DataFrame) -> int:
    return max(30, int(len(data) * 0.002))


def show_table(data: pd.DataFrame) -> pd.DataFrame:

    output = data.copy()

    if "Late_rate" in output.columns:
        output["Late_rate"] = (
            output["Late_rate"] * 100
        ).round(1)

        output = output.rename(
            columns={
                "Late_rate": "Late rate (%)"
            }
        )

    output = output.rename(
        columns={
            "Late_orders": "Late orders"
        }
    )

    if "Profit" in output.columns:
        output["Profit"] = output["Profit"].round(0)

    return output


# ============================================================
# 6. AGENT ANALYSIS FUNCTIONS
# ============================================================

@dataclass
class Result:
    title: str
    finding: str
    color: str = BRAND
    table: Optional[pd.DataFrame] = None
    fig: Optional[go.Figure] = None
    actions: list[str] = field(default_factory=list)


def tool_summary(data: pd.DataFrame) -> Result:

    total = len(data)

    if total == 0:
        return Result(
            "Situation summary",
            "No orders are available.",
            WARN,
        )

    late = data[
        data["Late_delivery_risk"] == 1
    ]

    messages = [
        f"**{len(late):,} of {total:,} orders "
        f"({len(late) / total:.1%})** are marked late."
    ]

    if "Shipping Mode" in data.columns:

        table = rate_table(data, "Shipping Mode")

        qualified = table[
            table["Orders"] >= min_volume(data)
        ]

        if not qualified.empty:

            worst = qualified.sort_values(
                "Late_rate",
                ascending=False,
            ).iloc[0]

            messages.append(
                f"**{worst['Shipping Mode']}** has the "
                f"highest late rate among modes with "
                f"sufficient volume ({worst['Late_rate']:.1%})."
            )

    if "Order Region" in data.columns and not late.empty:

        regions = (
            late["Order Region"]
            .dropna()
            .value_counts()
        )

        if not regions.empty:
            messages.append(
                f"**{regions.index[0]}** contributes "
                f"{regions.iloc[0]:,} late orders."
            )

    if not late.empty:
        messages.append(
            f"Recorded profit associated with late orders: "
            f"**{money(late['Profit'].sum())}**."
        )

    return Result(
        "Situation summary",
        " ".join(messages),
        BRAND,
    )


def tool_dimension(
    data: pd.DataFrame,
    dimension: str,
    label: str,
    ascending: bool = False,
) -> Result:

    if dimension not in data.columns:
        return Result(
            f"Risk by {label}",
            f"The dataset does not contain {dimension}.",
            WARN,
        )

    table = rate_table(data, dimension)

    qualified = table[
        table["Orders"] >= min_volume(data)
    ]

    ranked = (
        qualified
        if not qualified.empty
        else table
    )

    ranked = ranked.dropna(
        subset=[dimension]
    )

    if ranked.empty:
        return Result(
            f"Risk by {label}",
            "There is not enough usable data to rank this dimension.",
            WARN,
        )

    selected = ranked.sort_values(
        "Late_rate",
        ascending=ascending,
    ).iloc[0]

    most_orders = table.sort_values(
        "Late_orders",
        ascending=False,
    ).iloc[0]

    total_late = max(
        int(data["Late_delivery_risk"].sum()),
        1,
    )

    direction = (
        "lowest" if ascending else "highest"
    )

    finding = (
        f"**{selected[dimension]}** has the {direction} "
        f"late rate by {label}: "
        f"**{selected['Late_rate']:.1%}** across "
        f"{int(selected['Orders']):,} orders. "
        f"**{most_orders[dimension]}** has the most late orders: "
        f"{int(most_orders['Late_orders']):,}, representing "
        f"{most_orders['Late_orders'] / total_late:.1%} "
        f"of late orders."
    )

    chart_data = ranked.sort_values(
        "Late_rate",
        ascending=ascending,
    ).head(10).copy()

    chart_data[dimension] = (
        chart_data[dimension].astype(str)
    )

    fig = px.bar(
        chart_data,
        x="Late_rate",
        y=dimension,
        orientation="h",
        text=chart_data["Late_rate"].map(
            "{:.1%}".format
        ),
    )

    fig.update_traces(
        marker_color=[
            RISK if i == 0 else BRAND
            for i in range(len(chart_data))
        ],
        textposition="outside",
        cliponaxis=False,
    )

    fig.update_yaxes(
        autorange="reversed",
        title=None,
    )

    fig.update_xaxes(
        tickformat=".0%",
        title="Late-delivery rate",
    )

    actions = [
        f"Review operational processes for {selected[dimension]}."
    ]

    if dimension == "Shipping Mode":
        best = ranked.sort_values(
            "Late_rate"
        ).iloc[0]

        actions = [
            f"Review the carrier and promised delivery time "
            f"for {selected[dimension]}. Compare it with "
            f"{best[dimension]}, whose observed late rate is "
            f"{best['Late_rate']:.1%}."
        ]

    elif dimension in [
        "Category Name",
        "Department Name",
    ]:
        actions = [
            f"Audit packing quality and supplier lead times "
            f"for {selected[dimension]}."
        ]

    return Result(
        f"Late-delivery risk by {label}",
        finding,
        RISK,
        show_table(chart_data),
        fig,
        actions,
    )


def tool_root_cause(data: pd.DataFrame) -> Result:

    required = [
        "Shipping Mode",
        "Days for shipment (scheduled)",
        "Days for shipping (real)",
    ]

    late = data[
        data["Late_delivery_risk"] == 1
    ]

    if (
        late.empty
        or any(c not in data.columns for c in required)
        or data["Delay"].isna().all()
    ):
        return Result(
            "Root-cause analysis",
            "Actual and scheduled shipping data are needed "
            "to compare delivery delays.",
            WARN,
        )

    average_delay = late["Delay"].mean()

    grouped = data.groupby(
        "Shipping Mode",
        dropna=False,
    ).agg(
        Orders=("Late_delivery_risk", "size"),
        Late_rate=("Late_delivery_risk", "mean"),
        Late_orders=("Late_delivery_risk", "sum"),
        Promised_days=(
            "Days for shipment (scheduled)",
            "mean",
        ),
        Actual_days=(
            "Days for shipping (real)",
            "mean",
        ),
    ).reset_index()

    qualified = grouped[
        grouped["Orders"] >= min_volume(data)
    ]

    if not qualified.empty:
        grouped = qualified

    worst = grouped.sort_values(
        "Late_rate",
        ascending=False,
    ).iloc[0]

    share = (
        worst["Late_orders"]
        / max(len(late), 1)
    )

    finding = (
        f"Late orders arrive **{average_delay:.1f} days** "
        f"after the promised date on average. "
        f"**{worst['Shipping Mode']}** has a "
        f"{worst['Late_rate']:.1%} late rate. Its average "
        f"promised time is {worst['Promised_days']:.1f} days, "
        f"while average actual time is "
        f"{worst['Actual_days']:.1f} days. It accounts for "
        f"{share:.1%} of all late orders."
    )

    fig = go.Figure()

    fig.add_bar(
        name="Promised days",
        x=grouped["Shipping Mode"],
        y=grouped["Promised_days"],
        marker_color=OK,
    )

    fig.add_bar(
        name="Actual days",
        x=grouped["Shipping Mode"],
        y=grouped["Actual_days"],
        marker_color=RISK,
    )

    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Average days")

    table = grouped.rename(
        columns={
            "Promised_days": "Promised days",
            "Actual_days": "Actual days",
        }
    )

    table = show_table(table)

    actions = [
        f"Review the carrier handoff and delivery promise "
        f"for {worst['Shipping Mode']}.",
        "Flag short-promise orders for earlier carrier follow-up.",
    ]

    return Result(
        "Why orders are delayed",
        finding,
        RISK,
        table,
        fig,
        actions,
    )


def tool_financial(data: pd.DataFrame) -> Result:

    late = data[
        data["Late_delivery_risk"] == 1
    ]

    on_time = data[
        data["Late_delivery_risk"] == 0
    ]

    if late.empty:
        return Result(
            "Financial exposure",
            "No late orders are present in this filtered view.",
            OK,
        )

    late_loss = late.loc[
        late["Profit"] < 0,
        "Profit",
    ].sum()

    loss_count = int(
        (late["Profit"] < 0).sum()
    )

    late_average = late["Profit"].mean()

    ontime_average = (
        on_time["Profit"].mean()
        if not on_time.empty
        else np.nan
    )

    finding = (
        f"Late orders have recorded profit of "
        f"**{money(late['Profit'].sum())}**. "
        f"**{loss_count:,}** late orders are loss-making, "
        f"with combined recorded profit of "
        f"**{money(late_loss)}**. Average recorded profit "
        f"per late order is **{money(late_average)}**, "
        f"compared with **{money(ontime_average)}** "
        f"for on-time orders."
    )

    fig = go.Figure(
        go.Bar(
            x=["Late orders", "On-time orders"],
            y=[
                late_average,
                0 if pd.isna(ontime_average)
                else ontime_average,
            ],
            marker_color=[RISK, OK],
            text=[
                money(late_average),
                money(ontime_average),
            ],
            textposition="outside",
        )
    )

    fig.update_yaxes(
        title="Average recorded profit per order"
    )

    actions = [
        "Review discounts and fulfilment costs on loss-making late orders.",
        "Review service options for high-profit orders.",
    ]

    return Result(
        "Financial exposure",
        finding,
        WARN,
        None,
        fig,
        actions,
    )


def tool_trend(data: pd.DataFrame) -> Result:

    if (
        "order_date_parsed" not in data.columns
        or data["order_date_parsed"].notna().sum() == 0
    ):
        return Result(
            "Trend over time",
            "The dataset has no usable order-date column.",
            WARN,
        )

    dated = data.dropna(
        subset=["order_date_parsed"]
    ).copy()

    first = dated["order_date_parsed"].min()
    last = dated["order_date_parsed"].max()

    span_days = (last - first).days

    month_count = (
        dated["order_date_parsed"]
        .dt.to_period("M")
        .nunique()
    )

    if month_count >= 3:
        frequency, unit = "M", "month"
    elif span_days >= 14:
        frequency, unit = "W", "week"
    else:
        frequency, unit = "D", "day"

    dated["Period"] = (
        dated["order_date_parsed"]
        .dt.to_period(frequency)
        .astype(str)
    )

    grouped = dated.groupby(
        "Period"
    ).agg(
        Orders=("Late_delivery_risk", "size"),
        Late_rate=("Late_delivery_risk", "mean"),
    ).reset_index()

    grouped = grouped.sort_values("Period")
    grouped = grouped[
        grouped["Orders"] >= 10
    ]

    if len(grouped) < 2:
        return Result(
            "Trend over time",
            "There are not enough periods with at least "
            "10 orders to show a reliable trend. "
            "Clear some filters or upload the full dataset.",
            WARN,
        )

    high = grouped.loc[
        grouped["Late_rate"].idxmax()
    ]

    low = grouped.loc[
        grouped["Late_rate"].idxmin()
    ]

    change = (
        grouped["Late_rate"].iloc[-1]
        - grouped["Late_rate"].iloc[0]
    )

    finding = (
        f"Late rate peaked in **{high['Period']}** at "
        f"**{high['Late_rate']:.1%}** and was lowest in "
        f"**{low['Period']}** at **{low['Late_rate']:.1%}**. "
        f"From the first to last period, it changed by "
        f"**{change * 100:+.1f} percentage points**."
    )

    fig = px.line(
        grouped,
        x="Period",
        y="Late_rate",
        markers=True,
    )

    fig.update_traces(line_color=BRAND)
    fig.update_yaxes(
        tickformat=".0%",
        title="Late-delivery rate",
    )
    fig.update_xaxes(title=None)

    return Result(
        f"Late rate by {unit}",
        finding,
        BRAND,
        show_table(grouped.tail(12)),
        fig,
        [
            "Review capacity and operational buffers before recurring high-risk periods."
        ],
    )


def tool_recommend(data: pd.DataFrame) -> Result:

    rows = []
    minimum = min_volume(data)

    if "Shipping Mode" in data.columns:
        modes = rate_table(
            data,
            "Shipping Mode",
        )

        modes = modes[
            modes["Orders"] >= minimum
        ]

        if len(modes) >= 2:
            ordered = modes.sort_values(
                "Late_rate"
            )

            best = ordered.iloc[0]
            worst = ordered.iloc[-1]

            estimated = max(
                0,
                int(
                    worst["Orders"]
                    * (
                        worst["Late_rate"]
                        - best["Late_rate"]
                    )
                ),
            )

            rows.append(
                (
                    f"Review carrier or promise for {worst['Shipping Mode']}",
                    f"Observed late rate {worst['Late_rate']:.1%}, "
                    f"compared with {best['Late_rate']:.1%} "
                    f"for {best['Shipping Mode']}.",
                    f"Scenario only: approximately {estimated:,} "
                    "fewer late orders if the comparison rate were achieved.",
                )
            )

    late = data[
        data["Late_delivery_risk"] == 1
    ]

    for column, label in [
        ("Order Region", "region"),
        ("Order City", "city"),
        ("Category Name", "category"),
    ]:
        if column in data.columns and not late.empty:
            counts = (
                late[column]
                .dropna()
                .value_counts()
            )

            if not counts.empty:
                rows.append(
                    (
                        f"Prioritise {counts.index[0]} ({label})",
                        f"{counts.iloc[0]:,} late orders, "
                        f"{counts.iloc[0] / len(late):.1%} "
                        "of late orders.",
                        "Investigate the largest observed group of late orders.",
                    )
                )

    if not rows:
        return Result(
            "Recommended actions",
            "Not enough data in this view to rank actions.",
            WARN,
        )

    table = pd.DataFrame(
        rows,
        columns=[
            "Action",
            "Evidence",
            "Expected impact",
        ],
    )

    return Result(
        "Recommended actions",
        f"Top priority: **{rows[0][0]}**. Evidence: {rows[0][1]}",
        OK,
        table,
        None,
        [row[0] for row in rows[:3]],
    )


# ============================================================
# 7. QUESTION ROUTER
# ============================================================

def resolve_dimensions(
    question: str,
    columns: set[str],
) -> list[tuple[str, str]]:

    q = question.lower()
    dimensions = []

    def has(*words):
        return any(
            re.search(rf"\b{re.escape(word)}", q)
            for word in words
        )

    if has("city", "cities"):
        if "customer" in q and "Customer City" in columns:
            column = "Customer City"
        elif "Order City" in columns:
            column = "Order City"
        else:
            column = "Customer City"

        dimensions.append((column, "city"))

    if has("state"):
        if "customer" in q and "Customer State" in columns:
            column = "Customer State"
        elif "Order State" in columns:
            column = "Order State"
        else:
            column = "Customer State"

        dimensions.append((column, "state"))

    if has("country", "countries"):
        if "customer" in q and "Customer Country" in columns:
            column = "Customer Country"
        else:
            column = "Order Country"

        dimensions.append((column, "country"))

    if has("region"):
        dimensions.append(("Order Region", "region"))

    if has("market"):
        dimensions.append(("Market", "market"))

    if has("category", "categories", "product", "products"):
        dimensions.append(("Category Name", "category"))

    if has("department"):
        dimensions.append(("Department Name", "department"))

    if has("segment"):
        dimensions.append(("Customer Segment", "customer segment"))

    if has("shipping", "mode", "carrier"):
        dimensions.append(("Shipping Mode", "shipping mode"))

    if has("payment"):
        dimensions.append(("Type", "payment type"))

    return [
        (column, label)
        for column, label in dimensions
        if column in columns
    ]


def run_agent(
    question: str,
    data: pd.DataFrame,
) -> tuple[list[str], list[Result]]:

    q = question.lower()
    ascending = bool(
        re.search(
            r"\b(lowest|least|best|safest|minimum|min)\b",
            q,
        )
    )

    steps = []
    results = []

    for column, label in resolve_dimensions(
        q,
        set(data.columns),
    ):
        steps.append(
            f"Ranked late-delivery risk by {label}"
        )

        results.append(
            tool_dimension(
                data,
                column,
                label,
                ascending,
            )
        )

    if re.search(
        r"\b(why|reason|cause|root|bottleneck|delayed|delays|delay)\b",
        q,
    ):
        steps.append(
            "Compared promised and actual transit times"
        )
        results.append(tool_root_cause(data))

    if re.search(
        r"\b(profit|loss|benefit|money|financial|exposure|revenue|cost)\b",
        q,
    ):
        steps.append(
            "Reviewed recorded profit for late orders"
        )
        results.append(tool_financial(data))

    if re.search(
        r"\b(trend|month|monthly|time|seasonal)\b",
        q,
    ):
        steps.append(
            "Analysed late-delivery rates over time"
        )
        results.append(tool_trend(data))

    if re.search(
        r"\b(action|suggest|recommend|mitigate|fix|improve|reduce|solve|plan)\w*",
        q,
    ):
        steps.append(
            "Ranked recommended actions from observed patterns"
        )
        results.append(tool_recommend(data))

    if re.search(
        r"\b(summary|overview|status|brief|health)\b",
        q,
    ):
        steps.append(
            "Summarised the filtered orders"
        )
        results.insert(0, tool_summary(data))

    if not results:
        steps = [
            "No specific topic detected; running the full briefing"
        ]

        results = [
            tool_summary(data),
            tool_root_cause(data),
            tool_financial(data),
            tool_recommend(data),
        ]

    return steps, results


def render_result(
    result: Result,
    key: str,
) -> None:

    st.markdown(f"#### {result.title}")

    st.markdown(
        f"<div class='finding' style='--c:{result.color}'>"
        f"{safe_bold_html(result.finding)}</div>",
        unsafe_allow_html=True,
    )

    if result.fig is not None:
        st.plotly_chart(
            style_fig(result.fig, 300),
            key=f"chart-{key}",
            use_container_width=True,
        )

    if result.actions:
        st.markdown("**Recommended actions**")

        for action in result.actions:
            st.markdown(f"- {action}")

    if result.table is not None:
        with st.expander("View supporting data"):
            st.dataframe(
                result.table,
                hide_index=True,
                use_container_width=True,
            )


# ============================================================
# 8. LOAD DATA AND APPLY FILTERS
# ============================================================

with st.sidebar:
    st.markdown("## 🚚 Control Tower")
    st.markdown("### Data source")

    uploaded = st.file_uploader(
        "Upload supply-chain CSV",
        type=["csv"],
        help="You can also place the CSV beside app.py.",
    )

    st.markdown(
        f"[GitHub repository]({GITHUB_URL})  \n"
        f"[Hugging Face Space]({HF_URL})"
    )

try:
    if uploaded is not None:
        full = load_data(
            None,
            uploaded.getvalue(),
        )

        data_key = (
            f"upload-{uploaded.name}-{uploaded.size}"
        )

    else:
        default_path = find_default_file()

        if default_path is None:
            st.title("Supply Chain Control Tower")

            st.info(
                "No dataset found. Put "
                "`DataCoSupplyChainDataset.csv` beside "
                "`app.py`, or upload it using the sidebar."
            )

            st.stop()

        full = load_data(
            default_path,
            None,
        )

        data_key = (
            f"file-{default_path}-"
            f"{Path(default_path).stat().st_mtime}"
        )

except (
    ValueError,
    OSError,
    pd.errors.ParserError,
) as exc:

    st.error(f"Could not load the dataset: {exc}")
    st.stop()

if full.empty:
    st.error("No usable orders were found in the dataset.")
    st.stop()

view = full.copy()

with st.sidebar:
    st.markdown("### Filters")
    st.caption("Leave a filter empty to include all values.")

    filters = [
        ("Market", "Market"),
        ("Region", "Order Region"),
        ("Shipping mode", "Shipping Mode"),
        ("Customer segment", "Customer Segment"),
    ]

    for label, column in filters:
        if column in full.columns:
            options = sorted(
                full[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected = st.multiselect(
                label,
                options,
                placeholder="All",
                key=f"filter-{column}",
            )

            if selected:
                view = view[
                    view[column].astype(str).isin(selected)
                ]

if view.empty:
    st.warning(
        "No orders match these filters. Clear a filter to continue."
    )
    st.stop()


# ============================================================
# 9. HEADER AND KPIs
# ============================================================

st.markdown(
    "<p class='page-title'>Supply Chain Control Tower</p>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<p class='page-sub'>Late-delivery monitoring for "
    f"{len(view):,} filtered orders, with operational analysis "
    f"and a machine-learning risk predictor.</p>",
    unsafe_allow_html=True,
)

total_orders = len(view)

late_df = view[
    view["Late_delivery_risk"] == 1
]

late_count = len(late_df)

late_rate = (
    late_count / total_orders
    if total_orders
    else 0
)

average_days_late = (
    late_df["Delay"].mean()
    if late_count and late_df["Delay"].notna().any()
    else np.nan
)

k = st.columns(5)

k[0].markdown(
    kpi(
        "Total orders",
        f"{total_orders:,}",
        "After cancellation filtering",
    ),
    unsafe_allow_html=True,
)

k[1].markdown(
    kpi(
        "Late-delivery rate",
        f"{late_rate:.1%}",
        f"{late_count:,} late orders",
        RISK,
    ),
    unsafe_allow_html=True,
)

k[2].markdown(
    kpi(
        "Not flagged late",
        f"{total_orders - late_count:,}",
        f"{1 - late_rate:.1%} of orders",
        OK,
    ),
    unsafe_allow_html=True,
)

k[3].markdown(
    kpi(
        "Average days late",
        "n/a"
        if pd.isna(average_days_late)
        else f"{average_days_late:.1f}",
        "Late-flagged orders only",
    ),
    unsafe_allow_html=True,
)

k[4].markdown(
    kpi(
        "Profit on late orders",
        money(late_df["Profit"].sum()),
        f"{int((late_df['Profit'] < 0).sum()):,} loss-making",
    ),
    unsafe_allow_html=True,
)

tab_agent, tab_overview, tab_model = st.tabs(
    ["🤖 Agent", "📊 Overview", "🎯 Risk predictor"]
)


# ============================================================
# 10. AGENT TAB
# ============================================================

with tab_agent:

    st.markdown(
        "<div class='briefing-head'>"
        "<h3>Autopilot briefing</h3>"
        "<p>Analysis automatically updates with the selected filters.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    automatic_results = [
        tool_summary(view),
        tool_root_cause(view),
        tool_recommend(view),
    ]

    for index, result in enumerate(automatic_results):
        render_result(
            result,
            f"automatic-{index}",
        )

    st.divider()
    st.markdown("### Ask the agent")

    st.caption(
        "Ask a question in plain English. "
        "The agent chooses matching analysis functions."
    )

    examples = [
        "Which customer city has the highest late delivery risk?",
        "Why are orders delayed?",
        "How much profit is at risk?",
        "Which category is worst and what should we do?",
        "Show the monthly trend",
    ]

    example_columns = st.columns(len(examples))

    for column, example in zip(example_columns, examples):
        if column.button(
            example,
            key=f"example-{example}",
            use_container_width=True,
        ):
            st.session_state["agent_q"] = example

    question = st.text_input(
        "Your question",
        key="agent_q",
        placeholder=(
            "e.g. Which region has the most late orders? "
            "Suggest actions."
        ),
    )

    if question.strip():

        steps, results = run_agent(
            question,
            view,
        )

        with st.expander(
            "What the agent did",
            expanded=False,
        ):
            for index, step in enumerate(steps, start=1):
                st.markdown(f"{index}. {step}")

        for index, result in enumerate(results):
            render_result(
                result,
                f"question-{index}",
            )


# ============================================================
# 11. OVERVIEW TAB
# ============================================================

with tab_overview:

    left, right = st.columns(2)

    with left:
        st.markdown("#### Late rate by shipping mode")

        if "Shipping Mode" in view.columns:

            table = rate_table(
                view,
                "Shipping Mode",
            ).sort_values(
                "Late_rate",
                ascending=False,
            )

            fig = px.bar(
                table,
                x="Shipping Mode",
                y="Late_rate",
                text=table["Late_rate"].map(
                    "{:.1%}".format
                ),
            )

            fig.update_traces(
                marker_color=[
                    RISK if i == 0 else BRAND
                    for i in range(len(table))
                ],
                textposition="outside",
                cliponaxis=False,
            )

            fig.add_hline(
                y=late_rate,
                line_dash="dot",
                line_color=MUTED,
                annotation_text="Overall",
                annotation_position="top left",
            )

            fig.update_yaxes(
                tickformat=".0%",
                title=None,
                range=[
                    0,
                    min(
                        1.1,
                        max(
                            0.1,
                            table["Late_rate"].max() * 1.2,
                        ),
                    ),
                ],
            )

            fig.update_xaxes(title=None)

            st.plotly_chart(
                style_fig(fig),
                key="overview-shipping",
                use_container_width=True,
            )

    with right:
        st.markdown("#### Days late or early")

        if view["Delay"].notna().any():

            fig = px.histogram(
                view,
                x="Delay",
                color="Delivery Status",
                nbins=18,
                color_discrete_sequence=[
                    RISK,
                    OK,
                    BRAND,
                    WARN,
                ],
            )

            fig.update_layout(bargap=0.05)

            fig.update_xaxes(
                title="Actual minus promised days"
            )

            fig.update_yaxes(title="Orders")

            st.plotly_chart(
                style_fig(fig),
                key="overview-delay",
                use_container_width=True,
            )

        else:
            st.info(
                "Actual and scheduled shipping-day columns "
                "are not both available."
            )

    left2, right2 = st.columns(2)

    with left2:
        st.markdown("#### Late rate over time")

        trend_result = tool_trend(view)

        if trend_result.fig is not None:

            st.plotly_chart(
                style_fig(trend_result.fig),
                key="overview-trend",
                use_container_width=True,
            )

        else:
            st.info(trend_result.finding)

    with right2:
        st.markdown("#### Regions with the most late orders")

        if "Order Region" in view.columns:

            table = rate_table(
                view,
                "Order Region",
            ).sort_values(
                "Late_orders",
                ascending=False,
            ).head(10)

            fig = px.bar(
                table,
                x="Late_orders",
                y="Order Region",
                orientation="h",
                text="Late_orders",
            )

            fig.update_traces(
                marker_color=BRAND,
                textposition="outside",
                cliponaxis=False,
            )

            fig.update_yaxes(
                autorange="reversed",
                title=None,
            )

            fig.update_xaxes(title="Late orders")

            st.plotly_chart(
                style_fig(fig),
                key="overview-region",
                use_container_width=True,
            )

    st.markdown("#### Highest-delay orders")

    display_columns = [
        "Order Id",
        "Order City",
        "Order Region",
        "Category Name",
        "Shipping Mode",
        "Days for shipment (scheduled)",
        "Days for shipping (real)",
        "Delay",
        "Profit",
        "Delivery Status",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in view.columns
    ]

    if display_columns:

        late_orders = late_df.sort_values(
            "Delay",
            ascending=False,
            na_position="last",
        )[display_columns]

        st.dataframe(
            late_orders.head(100),
            hide_index=True,
            use_container_width=True,
            height=320,
        )

        st.download_button(
            "Download all filtered late orders (CSV)",
            data=late_orders.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="late_orders.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "No suitable order-detail columns were found."
        )


# ============================================================
# 12. RISK PREDICTOR TAB
# ============================================================

with tab_model:

    try:
        bundle = train_model(
            full,
            data_key,
        )

    except (ValueError, KeyError) as exc:
        st.error(
            f"Risk model could not be trained: {exc}"
        )

    else:

        st.markdown(
            "#### Model quality on a held-out test set"
        )

        metrics = st.columns(4)

        metrics[0].markdown(
            kpi(
                "Accuracy",
                f"{bundle['accuracy']:.1%}",
                f"{bundle['n_test']:,} test orders",
            ),
            unsafe_allow_html=True,
        )

        metrics[1].markdown(
            kpi(
                "Majority baseline",
                f"{bundle['baseline']:.1%}",
                "Always predicts the common class",
            ),
            unsafe_allow_html=True,
        )

        auc_text = (
            f"{bundle['auc']:.3f}"
            if pd.notna(bundle["auc"])
            else "n/a"
        )

        metrics[2].markdown(
            kpi(
                "ROC AUC",
                auc_text,
                "0.5 is random ranking",
            ),
            unsafe_allow_html=True,
        )

        metrics[3].markdown(
            kpi(
                "Recall for late orders",
                f"{bundle['recall']:.1%}",
                "Share of late orders detected",
            ),
            unsafe_allow_html=True,
        )

        st.caption(
            "The model uses order-time features such as shipping mode, "
            "promised shipping days, market, region, category, quantity "
            "and sales. Actual shipping days and the target label are "
            "excluded from the inputs. Metrics use a stratified random "
            "20% holdout split and do not guarantee future performance."
        )

        st.markdown(
            "#### Compare shipping options for a hypothetical order"
        )

        st.caption(
            "Select order attributes. Each shipping mode is scored "
            "using its typical promised shipping days in this dataset."
        )

        categorical_inputs = {}

        input_columns = st.columns(3)

        available_inputs = [
            column
            for column in [
                "Market",
                "Order Region",
                "Category Name",
                "Type",
                "Customer Segment",
                "Department Name",
            ]
            if column in bundle["cat"]
        ]

        for index, column in enumerate(available_inputs):

            options = sorted(
                full[column]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if not options:
                continue

            default_value = str(
                bundle["defaults"].get(
                    column,
                    options[0],
                )
            )

            default_index = (
                options.index(default_value)
                if default_value in options
                else 0
            )

            categorical_inputs[column] = (
                input_columns[index % 3].selectbox(
                    column,
                    options,
                    index=default_index,
                    key=f"predict-{column}",
                )
            )

        required_predictor_columns = [
            "Shipping Mode",
            "Days for shipment (scheduled)",
        ]

        if all(
            column in full.columns
            for column in required_predictor_columns
        ):

            scheduled_days = pd.to_numeric(
                full["Days for shipment (scheduled)"],
                errors="coerce",
            )

            mode_data = full.assign(
                _scheduled_numeric=scheduled_days
            )

            def typical_days(series):
                mode_values = series.mode()

                if not mode_values.empty:
                    return float(mode_values.iloc[0])

                return float(series.median())

            mode_days = (
                mode_data.groupby(
                    "Shipping Mode"
                )["_scheduled_numeric"]
                .agg(typical_days)
                .dropna()
            )

            candidate_rows = []

            for mode, days in mode_days.items():

                row = dict(bundle["defaults"])
                row.update(categorical_inputs)

                row["Shipping Mode"] = str(mode)

                row[
                    "Days for shipment (scheduled)"
                ] = float(days)

                candidate_rows.append(row)

            if candidate_rows:

                candidates = pd.DataFrame(
                    candidate_rows
                )

                for column in bundle["cat"]:
                    candidates[column] = (
                        candidates[column]
                        .astype("string")
                        .fillna("Unknown")
                        .astype(str)
                    )

                for column in bundle["num"]:
                    candidates[column] = pd.to_numeric(
                        candidates[column],
                        errors="coerce",
                    )

                candidates = candidates[
                    bundle["feature_cols"]
                ]

                probabilities = (
                    bundle["model"]
                    .predict_proba(candidates)[:, 1]
                )

                comparison = pd.DataFrame(
                    {
                        "Shipping Mode": [
                            row["Shipping Mode"]
                            for row in candidate_rows
                        ],
                        "Promised days": [
                            row[
                                "Days for shipment (scheduled)"
                            ]
                            for row in candidate_rows
                        ],
                        "Predicted late probability": probabilities,
                    }
                ).sort_values(
                    "Predicted late probability",
                    ascending=True,
                )

                best = comparison.iloc[0]

                message = (
                    f"**{best['Shipping Mode']}** has the lowest "
                    f"predicted late probability for these inputs: "
                    f"**{best['Predicted late probability']:.1%}**, "
                    f"with a typical {best['Promised days']:g}-day "
                    f"promise in this dataset."
                )

                st.markdown(
                    f"<div class='finding' style='--c:{OK}'>"
                    f"{safe_bold_html(message)}</div>",
                    unsafe_allow_html=True,
                )

                fig = px.bar(
                    comparison,
                    x="Predicted late probability",
                    y="Shipping Mode",
                    orientation="h",
                    text=comparison[
                        "Predicted late probability"
                    ].map("{:.1%}".format),
                )

                fig.update_traces(
                    marker_color=[
                        OK if index == 0 else BRAND
                        for index in range(len(comparison))
                    ],
                    textposition="outside",
                    cliponaxis=False,
                )

                fig.update_xaxes(
                    tickformat=".0%",
                    title="Predicted probability of late delivery",
                    range=[
                        0,
                        min(
                            1.1,
                            max(
                                0.1,
                                comparison[
                                    "Predicted late probability"
                                ].max() * 1.2,
                            ),
                        ),
                    ],
                )

                fig.update_yaxes(
                    title=None,
                    autorange="reversed",
                )

                st.plotly_chart(
                    style_fig(fig, 300),
                    key="predictor-comparison",
                    use_container_width=True,
                )

                st.dataframe(
                    comparison,
                    hide_index=True,
                    use_container_width=True,
                )

                st.caption(
                    "This is a model-based comparison, not a causal "
                    "guarantee. Validate operational feasibility before "
                    "changing a shipping service."
                )

            else:
                st.info(
                    "No usable shipping-mode and promised-day "
                    "combinations were found."
                )

        else:
            st.info(
                "The predictor needs Shipping Mode and "
                "Days for shipment (scheduled) columns."
            )