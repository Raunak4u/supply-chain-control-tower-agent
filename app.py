import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

st.set_page_config(page_title="Supply Chain Control Tower", layout="wide", initial_sidebar_state="expanded")

# --- SIDEBAR - No Hugging Face Link ---
st.sidebar.title("🚚 Control Tower")
st.sidebar.markdown("**Data Source**")

uploaded_file = st.sidebar.file_uploader("Upload your own CSV", type=["csv"], help="Apni DataCoSupplyChainDataset.csv ya koi bhi supply chain CSV upload karo")

st.sidebar.divider()
st.sidebar.success("✅ Ready to deploy")
st.sidebar.caption("Built by Raunak Kumar")

# --- MAIN ---
st.title("🚚 Supply Chain Control Tower")
st.caption("Agentic AI Supply Chain Risk Dashboard")

def load_data():
    # 1. Agar user ne file upload ki hai to wahi use karo
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file, encoding='latin1')
        except:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)

    # 2. Agar local me CSV hai to use karo
    if os.path.exists("DataCoSupplyChainDataset.csv"):
        try:
            return pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1')
        except:
            pass
    
    # 3. Nahi to dummy data
    np.random.seed(42)
    n=180
    return pd.DataFrame({
        'Type': np.random.choice(['DEBIT','TRANSFER','CASH','PAYMENT'], n),
        'Days for shipping (real)': np.random.randint(1,7,n),
        'Days for shipment (scheduled)': 4,
        'Benefit per order': np.random.uniform(-250, 300, n).round(2),
        'Sales per customer': np.random.uniform(100, 400, n).round(2),
        'Delivery Status': np.random.choice(['Advance shipping','Late delivery','Shipping on time'], n, p=[0.5,0.2,0.3]),
        'Late_delivery_risk': np.random.choice([0,1], n, p=[0.8,0.2]),
        'Category Name': 'Sporting Goods'
    })

df = load_data()

# KPIs
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Orders", len(df))
c2.metric("At Risk", int(df['Late_delivery_risk'].sum()) if 'Late_delivery_risk' in df.columns else 0)
c3.metric("Delayed", len(df[df['Delivery Status']=='Late delivery']) if 'Delivery Status' in df.columns else 0)
c4.metric("On Time", len(df[df['Delivery Status']!='Late delivery']) if 'Delivery Status' in df.columns else len(df))

if uploaded_file:
    st.success(f"✅ Your file loaded: {uploaded_file.name} - {len(df)} rows")
else:
    st.info(f"ℹ️ Using demo data - Loaded {len(df)} orders. Upload your CSV from sidebar to use your own data.")

col1, col2 = st.columns(2)
with col1:
    if 'Delivery Status' in df.columns:
        st.plotly_chart(px.histogram(df, x="Delivery Status", color="Delivery Status", title="📊 Order Overview"), use_container_width=True)
with col2:
    if 'Sales per customer' in df.columns and 'Benefit per order' in df.columns:
        st.plotly_chart(px.scatter(df, x="Sales per customer", y="Benefit per order", color="Late_delivery_risk" if 'Late_delivery_risk' in df.columns else None, title="💰 Risk vs Profit"), use_container_width=True)

st.subheader("📦 Data Preview")
st.dataframe(df.head(100), use_container_width=True)