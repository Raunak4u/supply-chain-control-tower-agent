import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Supply Chain Control Tower", layout="wide")
st.title("📦 Supply Chain Control Tower Agent")
st.caption("Agentic AI Supply Chain Risk Dashboard | Built by Raunak Kumar")

# Load data
try:
    df = pd.read_csv("test_orders.csv", encoding='latin-1')
    st.success(f"Loaded {len(df)} orders")

    # Calculate metrics - FIXED
    total = len(df)
    at_risk = df['Late_delivery_risk'].sum() if 'Late_delivery_risk' in df.columns else 0
    delayed = len(df[df['Delivery Status'] == 'Late delivery']) if 'Delivery Status' in df.columns else 0
    on_time = total - delayed

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", total)
    col2.metric("At Risk", int(at_risk))
    col3.metric("Delayed", delayed)
    col4.metric("On Time", on_time)

    st.subheader("📊 Order Overview")
    st.dataframe(df, use_container_width=True)

    if 'Late_delivery_risk' in df.columns:
        fig = px.histogram(df, x='Late_delivery_risk', title="Risk Score Distribution", color='Late_delivery_risk')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🤖 Agentic AI Analysis")
    query = st.text_input("Ask about supply chain risk:", placeholder="e.g. Which orders are at high risk?")
    if query:
        if at_risk > 0:
            st.warning(f"Found {int(at_risk)} high-risk orders! Recommend: Upgrade shipping to First Class.")
        else:
            st.success("All orders are low risk! No action needed.")
            # Show at risk orders
            if 'Late_delivery_risk' in df.columns:
                st.dataframe(df[df['Late_delivery_risk']==1])

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Make sure test_orders.csv is in same folder")

# Sidebar
st.sidebar.title("Controls")
st.sidebar.info("GitHub: Raunak4u/supply-chain-control-tower-agent\n\nHuggingFace: RAUNAK-KUMAR/supply-chain-control-tower-agent")