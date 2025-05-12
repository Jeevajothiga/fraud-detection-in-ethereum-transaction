import streamlit as st
import pandas as pd
import pytz
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import requests
from fraud_training_model import fetch_transactions, process_transactions, identify_fraud, train_fraud_model

# ----------------- Live Ethereum Price -----------------
@st.cache_data
def get_eth_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("ethereum", {}).get("usd", "N/A")
    return "N/A"

# ----------------- Page Config -----------------
st.set_page_config(page_title="Ethereum Fraud Detection", layout="wide")
st.markdown("<h1 style='text-align: center;'>🛡️ Ethereum Fraud Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

# ----------------- Input Section -----------------
wallet_address = st.text_input("🔎 Enter Ethereum Wallet Address", placeholder="0x...")

if "deploy_clicked" not in st.session_state:
    st.session_state.deploy_clicked = False

if st.button("🚀 Deploy Dashboard"):
    st.session_state.deploy_clicked = True

# ----------------- Main Dashboard -----------------
if st.session_state.deploy_clicked and wallet_address:
    model = train_fraud_model()
    if model:
        df_raw = fetch_transactions(wallet_address)
        if df_raw.empty:
            st.error("No transactions found.")
        else:
            df_raw = process_transactions(df_raw)
            df_raw = identify_fraud(df_raw)

            # Convert timestamps to IST safely
            ist = pytz.timezone('Asia/Kolkata')
            df_raw['timeStamp'] = pd.to_datetime(df_raw['timeStamp'], unit='s', errors='coerce')
            df_raw = df_raw.dropna(subset=['timeStamp'])
            if df_raw['timeStamp'].dt.tz is None:
                df_raw['timeStamp'] = df_raw['timeStamp'].dt.tz_localize('UTC')
            df_raw['timeStamp'] = df_raw['timeStamp'].dt.tz_convert(ist)

            # Wallet Status & ETH Price
            last_tx_time = df_raw['timeStamp'].max()
            days_since_last_tx = (pd.Timestamp.utcnow().replace(tzinfo=pytz.UTC).astimezone(ist) - last_tx_time).days
            wallet_status = "🟢 Active" if days_since_last_tx < 180 else "🔴 Inactive"
            eth_price = get_eth_price()

            st.markdown(f"""
                <div style="padding:10px;margin-bottom:15px;border-radius:8px;">
                    <b>Wallet Status:</b> {wallet_status} &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Last Transaction:</b> {last_tx_time.strftime('%d-%m-%Y %I:%M %p')} IST &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>Days Since:</b> {days_since_last_tx} &nbsp;&nbsp;|&nbsp;&nbsp;
                    <b>💱 Live ETH Price:</b> ${eth_price}
                </div>
            """, unsafe_allow_html=True)

            # ----------------- Transaction Table -----------------
            st.markdown("## 📋 Full Transaction Table")
            df_raw['formatted_time'] = df_raw['timeStamp'].dt.strftime('%d-%m-%Y %I:%M %p')
            available_cols = df_raw.columns.tolist()
            display_cols = ['hash', 'from', 'to', 'value', 'formatted_time']
            if 'fraud_type' in available_cols:
                display_cols.append('fraud_type')
            if 'risk_score' in available_cols:
                display_cols.append('risk_score')
            if 'reason' in available_cols:
                display_cols.append('reason')

            st.dataframe(df_raw[display_cols].rename(columns={'formatted_time': 'Timestamp'}), use_container_width=True)
            st.markdown("---")

            # ----------------- Filters -----------------
            st.markdown("## 🎛️ Visualization Filters")
            max_eth = float(df_raw['value'].max())
            col1, col2, col3 = st.columns([3, 3, 2])
            with col1:
                if max_eth > 0:
                    amount_range = st.slider("💸 Filter by Value (ETH)", 0.0, max_eth, (0.0, max_eth), step=max(0.1, max_eth / 100))
                else:
                    st.warning("No transaction values found to filter.")
                    amount_range = (0.0, 0.0)
            with col2:
                all_fraud_types = [
                    "Ponzi Scheme", "Money Laundering", "Pump & Dump", "Rug Pull",
                    "Phishing Scam", "Flash Loan Attack", "Bot Activity", "Address Poisoning",
                    "Wash Trading", "Cyclic Transactions", "Honeypot Scam", "Fake Airdrop"
                ]
                fraud_types = ["All"] + sorted(all_fraud_types)
                fraud_filter = st.selectbox("🔍 Filter by Fraud Type", fraud_types)
            with col3:
                start_date = df_raw['timeStamp'].dt.date.min()
                end_date = df_raw['timeStamp'].dt.date.max()
                date_range = st.date_input("📅 Filter by Date Range", (start_date, end_date))

            # ----------------- Apply Filters -----------------
            df_filtered = df_raw[
                (df_raw['value'] >= amount_range[0]) & (df_raw['value'] <= amount_range[1])
            ]
            if isinstance(date_range, tuple) and len(date_range) == 2:
                df_filtered = df_filtered[
                    (df_filtered['timeStamp'].dt.date >= date_range[0]) &
                    (df_filtered['timeStamp'].dt.date <= date_range[1])
                ]
            if fraud_filter != "All" and 'fraud_type' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['fraud_type'] == fraud_filter]

            # ----------------- Fraud Summary -----------------
            fraud_cases = df_filtered[df_filtered.get('is_fraud') == 1] if 'is_fraud' in df_filtered.columns else pd.DataFrame()
            st.markdown(f"**💰 Filtered Transactions:** {len(df_filtered)} | 🚨 Fraud Cases: {len(fraud_cases)}")
            if not df_filtered.empty:
                st.markdown(f"💎 **Highest Amount Transferred:** `{df_filtered['value'].max():,.4f}` ETH")
            if not fraud_cases.empty and 'fraud_type' in fraud_cases.columns:
                st.markdown(f"**Detected Fraud Types:** {fraud_cases['fraud_type'].value_counts().to_dict()}")

            # ----------------- Network Graph -----------------
            st.markdown("## 🌐 Transaction Network Graph")
            if not df_filtered.empty:
                G = nx.from_pandas_edgelist(df_filtered, source='from', target='to', edge_attr=True, create_using=nx.DiGraph())
                pos = nx.spring_layout(G, k=0.5, seed=42)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), mode='lines', hoverinfo='none')
                node_x, node_y, node_text = [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(str(node))
                node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text,
                                        marker=dict(color='skyblue', size=10, line=dict(width=2)), hoverinfo='text')
                fig_net = go.Figure(data=[edge_trace, node_trace],
                                    layout=go.Layout(
                                                    title=dict(text='Transaction Network', font=dict(size=16)),
                                                    margin=dict(b=20, l=5, r=5, t=40),
                                                    xaxis=dict(showgrid=False, zeroline=False),
                                                    yaxis=dict(showgrid=False, zeroline=False)
                                                    )

                st.plotly_chart(fig_net, use_container_width=True)
            else:
                st.info("No data available for network graph.")
            st.markdown("---")

            # ----------------- Bar Chart -----------------
            st.markdown("## 📊 Transactions Over Time")
            if not df_filtered.empty:
                df_filtered['date'] = df_filtered['timeStamp'].dt.date
                tx_counts = df_filtered.groupby('date').size().reset_index(name='count')
                fig_bar = px.bar(tx_counts, x='date', y='count', color='count',
                                 labels={'date': 'Date', 'count': 'Transactions'})
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No transactions to show in this range.")
            st.markdown("---")

            # ----------------- Line Chart -----------------
            st.markdown("## 📈 Fraud Trend Over Time")
            if not fraud_cases.empty:
                fraud_counts = fraud_cases.groupby(fraud_cases['timeStamp'].dt.date).size().reset_index(name='count')
                fraud_counts['timeStamp'] = pd.to_datetime(fraud_counts['timeStamp'])
                fig_line = px.line(fraud_counts, x='timeStamp', y='count', markers=True, title='Fraud Over Time')
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("✅ No fraud cases detected in selected range.")

            # ----------------- Export Button -----------------
            st.markdown("---")
            if not fraud_cases.empty:
                st.download_button(
                    label="📥 Download Fraud Report (CSV)",
                    data=fraud_cases.to_csv(index=False),
                    file_name=f"fraud_report_{wallet_address[:6]}.csv",
                    mime="text/csv"
                )
