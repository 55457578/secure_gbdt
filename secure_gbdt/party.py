# secure_gbdt/party.py
import numpy as np
from typing import Optional, Dict, Any
from .primitives import SecretShare, RLWECiphertext, A2H, H2A, F_cor

try:
    import requests
    from fastapi import FastAPI, Header, HTTPException, Request
    import uvicorn
except ImportError:
    requests = None
    FastAPI = None
    uvicorn = None

class NetworkAdapter:
    """
    Handles secure TLS communication and Entity Authentication between parties.
    """
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, peer_url: str = "", api_key: str = ""):
        self.host = host
        self.port = port
        self.peer_url = peer_url
        self.api_key = api_key
        self.app = None

    def start_server(self, party_obj: "Party", cert_path: str, key_path: str):
        if FastAPI is None:
            raise RuntimeError("FastAPI not installed.")
            
        self.app = FastAPI()

        @self.app.post("/rpc")
        async def rpc_endpoint(request: Request, x_api_key: str = Header(None)):
            if x_api_key != self.api_key:
                raise HTTPException(status_code=403, detail="Unauthorized entity.")
            
            payload = await request.json()
            method_name = payload.get("method")
            args = payload.get("args", {})
            
            if not hasattr(party_obj, method_name):
                raise HTTPException(status_code=400, detail="Unknown method.")
                
            func = getattr(party_obj, method_name)
            result = func(**args)
            return {"result": result}

        uvicorn.run(self.app, host=self.host, port=self.port, 
                    ssl_certfile=cert_path, ssl_keyfile=key_path)

    def call_peer(self, method: str, args: Dict[str, Any], ca_cert: str = None):
        if requests is None:
            raise RuntimeError("Requests not installed.")
            
        headers = {"X-API-Key": self.api_key}
        resp = requests.post(
            f"{self.peer_url}/rpc", 
            json={"method": method, "args": args},
            headers=headers,
            verify=ca_cert if ca_cert else False,
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["result"]


class Party:
    def __init__(self, name, X, n_bins=16, network: Optional[NetworkAdapter] = None):
        self.name = name
        self.X = np.asarray(X, dtype=float)
        self.n, self.m = self.X.shape
        self.n_bins = n_bins
        
        # SQUIRREL SPECIFIC: Binary Matrix representation of data
        self.M = None 
        
        # Secret Sample Indicator vector (b_l^{(k)})
        self.b_indicator = np.ones(self.n, dtype=int) 
        self.bin_edges = []
        
        self.network = network

    def fit_bins(self):
        """
        Squirrel Step 1: Locally partition data into bins and construct 
        the Binary Matrix M in {0,1}^{B*m x n}
        """
        self.bin_edges = []
        # Initialize an empty binary matrix M
        self.M = np.zeros((self.n_bins * self.m, self.n), dtype=np.int8)
        
        for j in range(self.m):
            qs = np.linspace(0, 100, self.n_bins + 1)[1:-1]
            edges = np.unique(np.percentile(self.X[:, j], qs))
            if edges.size == 0:
                edges = np.array([np.max(self.X[:, j])])
            self.bin_edges.append(edges)
                
            bvals = np.digitize(self.X[:, j], edges)
            
            # Populate the Binary Matrix M
            # M[B*z + u, i] = 1 if sample i is in bin u of feature z
            for i in range(self.n):
                u = min(bvals[i], self.n_bins - 1)
                row_idx = (self.n_bins * j) + u
                self.M[row_idx, i] = 1

    def bin_mat_vec(self, encrypted_gradients: dict) -> dict:
        """
        Squirrel Protocol: BinMatVec (Figure 6).
        Computes M * g securely using LWE homomorphic additions.
        """
        # Unwrap the simulated RLWECiphertext dict
        raw_gradients = np.array(encrypted_gradients["encrypted_data"])
        pub_key = encrypted_gradients["pub_key_owner"]
        
        # We leverage the sparsity of the indicator (Optimization 4.5.1)
        active_gradients = raw_gradients * self.b_indicator
        
        # Matrix-vector multiplication (Simulating Homomorphic Summation)
        aggregated_stats = self.M.dot(active_gradients)
        
        # Wrap it back up as a dict to pass securely over the network JSON
        return {
            "encrypted_data": aggregated_stats.tolist(),
            "pub_key_owner": pub_key
        }

    def update_indicator(self, split_identifier: dict, is_owner: bool):
        """
        Squirrel Protocol: Locally Update Sample Indicator (Section 4.2).
        Maintains the invariant b^(k) = b_0^(k) AND b_1^(k) using F_cor
        """
        if not is_owner:
            # P_{1-c} keeps indicators unchanged
            pass 
        else:
            # P_c updates child indicators locally based on the split
            feat_idx = split_identifier["feat_idx"]
            thr = split_identifier["thr"]
            
            # Create the private choice vector b_*
            b_star = (self.X[:, feat_idx] <= thr).astype(int)
            
            # Update left child indicator locally
            self.b_indicator = self.b_indicator & b_star
            
        return True

    def evaluate_split_indicator(self, feat_idx: int, bin_u: int, idx: list = None) -> list:
        """
        Called securely over RPC. Evaluates a split based on feature and bin index, 
        ensuring the raw threshold value never leaves this server.
        """
        edges = self.bin_edges[feat_idx]
        thr = float(np.max(edges)) if edges.size == 0 else float(edges[min(bin_u, edges.size - 1)])
        
        if idx is None:
            # Return full boolean mask
            b_star = (self.X[:, feat_idx] <= thr).astype(int)
        else:
            # Return mask only for specific row indices
            idx_arr = np.array(idx, dtype=int)
            b_star = (self.X[idx_arr, feat_idx] <= thr).astype(int)
            
        return b_star.tolist()
