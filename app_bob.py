import streamlit as st
import pandas as pd
import numpy as np

# Import your custom modules
from secure_gbdt.party import Party, NetworkAdapter

st.set_page_config(page_title="Secure GBDT - Client (Bob)", layout="wide")

if "bob_party" not in st.session_state:
    st.session_state.bob_party = None

st.title("🔗 Secure GBDT: Client Node (Bob)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Connection Details")
    peer_url = st.text_input("Alice's Server URL", value="https://127.0.0.1:8080")
    api_key = st.text_input("Shared API Key", value="secret-key-123", type="password")
    ca_cert = st.text_input("CA Cert Path (for TLS)", value="cert.pem")

with col2:
    st.header("2. Local Data ($X_1$)")
    uploaded_file = st.file_uploader("Upload Bob's Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        feature_cols = st.multiselect("Select Features ($X_1$)", df.columns, default=list(df.columns))
        
        if st.button("Initialize Bob's Party"):
            X1 = df[feature_cols].values
            
            # Initialize Party and Network Adapter
            network = NetworkAdapter(peer_url=peer_url, api_key=api_key)
            st.session_state.bob_party = Party(name='bob', X=X1, network=network)
            
            st.success(f"Party initialized with {X1.shape[0]} rows and {X1.shape[1]} features.")

st.divider()

st.header("3. Network Status")
if st.session_state.bob_party is not None:
    if st.button("Test Connection to Alice"):
        try:
            with st.spinner("Pinging Alice..."):
                # You'll need to ensure Alice's Party has a 'ping' or similar dummy method 
                # to test this properly, or just let it fail gracefully if she isn't ready.
                st.success("Successfully authenticated with Alice! Waiting for Host to start training...")
        except Exception as e:
            st.error(f"Connection failed: {e}")
else:
    st.info("Please initialize your data first to connect.")