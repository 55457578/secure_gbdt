# secure_gbdt/party.py
import numpy as np
from typing import Optional, Dict, Any

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
        # For local testing without strict TLS, verify can be False. 
        # In production, pass a valid ca_cert.
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
        self.bin_edges = []         
        self.binned = []            
        self.network = network

    def fit_bins(self):
        self.bin_edges = []
        self.binned = []
        for j in range(self.m):
            qs = np.linspace(0, 100, self.n_bins + 1)[1:-1]
            edges = np.unique(np.percentile(self.X[:, j], qs))
            if edges.size == 0:
                edges = np.array([np.max(self.X[:, j])])
            b = np.digitize(self.X[:, j], edges) 
            self.bin_edges.append(edges)
            self.binned.append(b)
            
    # --- NEW FEDERATED METHODS FOR RPC ---

    def get_best_split(self, idx: list, g: list, h: list, gamma: float, epsilon: float) -> dict:
        """
        Evaluates local features to find the best split based on gradients.
        Returns JSON-serializable dictionary.
        """
        # Convert RPC lists back to numpy arrays
        idx = np.array(idx, dtype=int)
        g = np.array(g, dtype=float)
        h = np.array(h, dtype=float)
        
        from .primitives import add_dp_noise
        
        best_gain = -1e18
        best = None
        
        for j in range(self.m):
            bvals = self.binned[j][idx]
            if bvals.size == 0:
                continue
            maxb = int(np.max(bvals))
            B_eff = max(self.n_bins, maxb + 1)
            
            G = np.bincount(bvals, weights=g, minlength=B_eff).astype(float)[:self.n_bins]
            H = np.bincount(bvals, weights=h, minlength=B_eff).astype(float)[:self.n_bins]

            G_noisy = np.array([add_dp_noise(val, epsilon) for val in G])
            H_noisy = np.array([add_dp_noise(val, epsilon) for val in H])

            G_cum = np.cumsum(G_noisy)
            H_cum = np.cumsum(H_noisy)
            G_tot = G_noisy.sum()
            H_tot = H_noisy.sum()

            for u in range(0, self.n_bins - 1):
                GL, HL = G_cum[u], H_cum[u]
                GR, HR = G_tot - GL, H_tot - HL
                
                gain = (GL * GL) / (gamma + HL + 1e-12) + \
                       (GR * GR) / (gamma + HR + 1e-12) - \
                       (G_tot * G_tot) / (gamma + H_tot + 1e-12)
                       
                if np.isfinite(gain) and gain > best_gain:
                    edges = self.bin_edges[j]
                    thr = float(np.max(self.X[:, j])) if edges.size == 0 else float(edges[int(max(0, min(u, edges.size - 1)))])
                    best_gain = gain
                    best = (j, u, thr)

        if best is None:
            return {"gain": float(-1e18), "feat_idx": -1, "bin_u": -1, "thr": 0.0}
            
        return {
            "gain": float(best_gain), 
            "feat_idx": int(best[0]), 
            "bin_u": int(best[1]), 
            "thr": float(best[2])
        }

    def evaluate_split(self, idx: list, feat_idx: int, thr: float) -> list:
        """
        Returns a boolean mask (as a list) for routing data down the tree.
        """
        idx = np.array(idx, dtype=int)
        sel = (self.X[idx, feat_idx] <= thr)
        return sel.tolist()