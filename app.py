import streamlit as st
import pandas as pd
import plotly.express as px

# Import your tools if available
try:
    import tools
except:
    pass

st.set_page_config(page_title="Supply Chain Control Tower", page_icon="📦", layout="wide")

st.title("📦 Supply Chain Control Tower Agent")
st.markdown("**Agentic AI Supply Chain Risk Dashboard** | Built by Raunak Kumar")

# Load data
try:
    df = pd.read_csv("test_orders.csv")
    st.success(f"Loaded {len(df)} orders")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", len(df))
    col2.metric("At Risk", len(df[df['risk_score'] > 70]) if 'risk_score' in df.columns else "N/A")
    col3.metric("Delayed", len(df[df['status'] == 'Delayed']) if 'status' in df.columns else "N/A")
    col4.metric("On Time", len(df[df['status'] == 'On Time']) if 'status' in df.columns else "N/A")

    st.subheader("📊 Order Overview")
    st.dataframe(df, use_container_width=True)
    
    if 'risk_score' in df.columns:
        fig = px.histogram(df, x="risk_score", title="Risk Score Distribution")
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Showing demo data - {e}")
    # Fallback demo
    demo = pd.DataFrame({
        "Order ID": ["ORD001", "ORD002", "ORD003"],
        "Origin": ["Mumbai", "Delhi", "Chennai"],
        "Destination": ["Dubai", "Singapore", "USA"],
        "Risk Score": [85, 45, 92],
        "Status": ["At Risk", "On Time", "Delayed"]
    })
    st.dataframe(demo, use_container_width=True)

st.subheader("🤖 Agentic AI Analysis")
user_query = st.text_input("Ask about supply chain risk (e.g., Which orders are at high risk?)")
if user_query:
    with st.spinner("Analyzing..."):
        st.info(f"**AI Agent Response for:** '{user_query}'\n\nBased on test_orders.csv, 2 orders are high risk due to geopolitical and weather disruptions. Recommended action: Reroute via alternate hub and notify supplier. (This is demo logic - connect your LLM here)")

st.sidebar.title("Controls")
st.sidebar.info("GitHub: Raunak4u/supply-chain-control-tower-agent\nHuggingFace: RAUNAK-KUMAR/supply-chain-control-tower-agent")