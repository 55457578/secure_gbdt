import streamlit as st
import pandas as pd
import numpy as np
import threading
import time

# Import your custom modules
from secure_gbdt.party import Party, NetworkAdapter
from secure_gbdt.vertical_gbdt import VerticalGBDT

st.set_page_config(page_title="Secure GBDT - Alice (Host)", layout="wide")

# --- STATE MANAGEMENT ---
if "alice_party" not in st.session_state:
    st.session_state.alice_party = None
if "server_running" not in st.session_state:
    st.session_state.server_running = False

def start_background_server(party, host, port, api_key, cert_path, key_path):
    """Runs the FastAPI server in a separate thread to prevent UI freezing."""
    network = NetworkAdapter(host=host, port=port, api_key=api_key)
    # This will block the thread, which is fine since it's a background thread
    network.start_server(party, cert_path=cert_path, key_path=key_path)

# --- UI LAYOUT ---
st.title("🛡️ Secure GBDT: Host Control Center (Alice)")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Network Configuration")
    host_ip = st.text_input("Host IP", value="0.0.0.0")
    port = st.number_input("Port", value=8080, step=1)
    api_key = st.text_input("Shared API Key", value="secret-key-123", type="password")
    
    st.markdown("*(Requires existing TLS certificates)*")
    cert_path = st.text_input("Cert Path", value="cert.pem")
    key_path = st.text_input("Key Path", value="key.pem")

with col2:
    st.header("2. Local Data ($X_0$ and $y$)")
    uploaded_file = st.file_uploader("Upload Alice's Dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        # Assume the last column is the target 'y' for this example
        feature_cols = st.multiselect("Select Features ($X_0$)", df.columns[:-1], default=list(df.columns[:-1]))
        target_col = st.selectbox("Select Target ($y$)", df.columns, index=len(df.columns)-1)
        
        if st.button("Initialize Alice's Party"):
            X0 = df[feature_cols].values
            y = df[target_col].values
            
            # Store in session state
            st.session_state.alice_party = Party(name='alice', X=X0)
            st.session_state.y_labels = y
            st.success(f"Party initialized with {X0.shape[0]} rows and {X0.shape[1]} features.")

st.divider()

# --- SERVER & TRAINING CONTROLS ---
st.header("3. Execution")

col3, col4 = st.columns(2)

with col3:
    if st.button("Start RPC Server", disabled=st.session_state.alice_party is None or st.session_state.server_running):
        # Start server in a background daemon thread
        server_thread = threading.Thread(
            target=start_background_server, 
            args=(st.session_state.alice_party, host_ip, port, api_key, cert_path, key_path),
            daemon=True
        )
        server_thread.start()
        st.session_state.server_running = True
        st.rerun()

    if st.session_state.server_running:
        st.success(f"✅ Server listening on {host_ip}:{port}. Waiting for Bob...")

with col4:
    st.subheader("Hyperparameters")
    max_depth = st.slider("Max Depth", 1, 10, 4)
    epsilon = st.slider("DP Epsilon (Privacy Budget)", 0.1, 10.0, 0.5)
    
    if st.button("Start Training Process", disabled=not st.session_state.server_running):
        with st.spinner("Coordinating with Bob and building trees..."):
            # NOTE: Your VerticalGBDT class currently expects Bob's data locally.
            # You will eventually replace this block with the network-coordinated training loop.
            st.info("Initiating tree building protocol...")
            time.sleep(2) # Mocking network delay
            st.success("Training Complete! Forest generated.")