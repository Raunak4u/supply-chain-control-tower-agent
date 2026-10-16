"""smolagents tools used by the Supply Chain Control Tower Agent."""
from smolagents import tool
import joblib
import pandas as pd

# The Gradio app sets this DataFrame for each uploaded file.
CURRENT_ORDERS = None
MODEL_PATH = "model.pkl"

def set_current_orders(df):
    global CURRENT_ORDERS
    CURRENT_ORDERS = df.copy()

def _normalized_orders():
    if CURRENT_ORDERS is None:
        return None
    df = CURRENT_ORDERS.copy()
    if "Category" not in df.columns and "Category Name" in df.columns:
        df = df.rename(columns={"Category Name": "Category"})
    if "Order Id" not in df.columns and "Order ID" in df.columns:
        df = df.rename(columns={"Order ID": "Order Id"})
    return df

@tool
def late_delivery_predictor(order_id: str) -> str:
    """Predict whether a supplied order is at high risk of late delivery.
    Args:
        order_id: Order Id value from the uploaded orders CSV.
    """
    df = _normalized_orders()
    if df is None:
        return "ERROR: Upload an orders CSV first."
    if "Order Id" not in df.columns:
        return "ERROR: CSV must include an 'Order Id' column."
    row = df[df["Order Id"].astype(str) == str(order_id)]
    if row.empty:
        return f"ERROR: Order ID {order_id} was not found in the uploaded file."
    row = row.iloc[[0]].copy()
    required = ["Days for shipping (real)", "Days for shipment (scheduled)",
                "Order Status", "Shipping Mode"]
    missing = [c for c in required if c not in row.columns]
    if missing:
        return f"ERROR: Missing columns required for prediction: {missing}"
    if "Category" not in row.columns:
        return "ERROR: CSV needs 'Category' or 'Category Name'."
    if "model.pkl" == MODEL_PATH:
        try:
            bundle = joblib.load(MODEL_PATH)
        except FileNotFoundError:
            return "ERROR: model.pkl not found. Run train_model.py first."
    else:
        bundle = joblib.load(MODEL_PATH)
    delay = (pd.to_numeric(row["Days for shipping (real)"], errors="coerce").iloc[0]
             - pd.to_numeric(row["Days for shipment (scheduled)"], errors="coerce").iloc[0])
    features = pd.DataFrame([{
        "Delay": delay,
        "Order Status": str(row["Order Status"].iloc[0]) if pd.notna(row["Order Status"].iloc[0]) else "Unknown",
        "Shipping Mode": str(row["Shipping Mode"].iloc[0]) if pd.notna(row["Shipping Mode"].iloc[0]) else "Unknown",
        "Category": str(row["Category"].iloc[0]) if pd.notna(row["Category"].iloc[0]) else "Unknown",
    }])
    prediction = int(bundle["model"].predict(features)[0])
    risk = "HIGH RISK (1)" if prediction == 1 else "LOW RISK (0)"
    return (f"Order {order_id}: {risk}. Calculated Delay = {delay} days "
            f"(actual shipping days minus scheduled days); Shipping Mode = "
            f"{features['Shipping Mode'].iloc[0]}; Category = {features['Category'].iloc[0]}.")

@tool
def alternative_shipping_suggester(current_mode: str) -> str:
    """Suggest a faster shipping class when the current mode is delayed.
    Args:
        current_mode: Current shipping method, such as Standard Class.
    """
    mode = (current_mode or "").strip().lower()
    if mode == "standard class":
        return ("Suggested action: upgrade to Second Class or First Class. "
                "Using the dataset's usual scheduled-day tiers, Second Class can save "
                "about 2 days and First Class about 3 days versus Standard Class. "
                "Confirm carrier SLA, price, and destination before changing the order.")
    if mode == "second class":
        return ("Suggested action: consider First Class. It may save about 1 day "
                "versus Second Class; confirm carrier SLA, price, and destination.")
    if mode == "first class":
        return ("First Class is already the fastest listed class. Escalate with the carrier "
                "and check fulfillment/dispatch bottlenecks.")
    return ("Shipping mode not recognized. Compare Second Class and First Class against "
            "the current service's actual SLA, cost, and destination before upgrading.")

@tool
def inventory_impact_checker(category: str) -> str:
    """Return a simple placeholder inventory impact note for a product category.
    Args:
        category: Product category for the order.
    """
    return f"Electronics stock low due to delay. Category checked: {category}. (Dummy inventory rule.)"
