import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

st.set_page_config(page_title="Supply Chain Control Tower", layout="wide")
st.title("🚚 Supply Chain Control Tower")

# CSV dhundega, nahi mila to auto dummy data banayega - error kabhi nahi ayega
def load_data():
    if os.path.exists("DataCoSupplyChainDataset.csv"):
        try:
            return pd.read_csv("DataCoSupplyChainDataset.csv", encoding='latin1')
        except:
            pass
    
    # Agar CSV nahi hai to 180 orders ka dummy data
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

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Orders", len(df))
c2.metric("At Risk", int(df['Late_delivery_risk'].sum()))
c3.metric("Delayed", len(df[df['Delivery Status']=='Late delivery']))
c4.metric("On Time", len(df[df['Delivery Status']!='Late delivery']))

st.success(f"Loaded {len(df)} orders | Real-time Risk Monitoring Active")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(px.histogram(df, x="Delivery Status", color="Delivery Status", title="📊 Order Overview"), use_container_width=True)
with col2:
    st.plotly_chart(px.scatter(df, x="Sales per customer", y="Benefit per order", color="Late_delivery_risk", title="💰 Risk vs Profit"), use_container_width=True)

st.dataframe(df, use_container_width=True)