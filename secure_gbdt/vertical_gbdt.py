# secure_gbdt/vertical_gbdt.py
import numpy as np
from .nodes import Node
from .primitives import SecretShare, RLWECiphertext, A2H, H2A, seg3_sigmoid

class VerticalGBDT:
    def __init__(self, host_party, y, max_depth=4, n_trees=10,
                 min_samples_leaf=10, gamma=1e-3, learning_rate=0.1):
        """
        Squirrel-compliant Coordinator.
        """
        self.host = host_party
        self.y = np.asarray(y, dtype=float)
        self.n = self.y.shape[0]
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.forest = []

    def fit(self):
        # 1. Local binning (Constructs Binary Matrix M locally)
        self.host.fit_bins()
        
        # 2. Instruct Client to bin features
        if self.host.network:
            print("[Network] Instructing Client to construct Binary Matrix M...")
            self.host.network.call_peer("fit_bins", args={})
            
        y_pred = np.zeros(self.n, dtype=float)

        for tree_i in range(self.n_trees):
            print(f"Building Tree {tree_i + 1}/{self.n_trees}")
            
            # --- SQUIRREL PROTOCOL: 1st and 2nd order gradients via Seg3Sigmoid ---
            # In true Squirrel, y_pred is a secret share. We wrap it here for the simulation.
            y_pred_share = SecretShare(y_pred, party_id=self.host.name)
            p = seg3_sigmoid(y_pred_share) 
            
            g = p - self.y
            h = p * (1.0 - p)

            idx_all = np.arange(self.n, dtype=int)
            tree = self._build_tree(idx_all, g, h, depth=0)
            self.forest.append(tree)
            
            self._update_predictions(tree, y_pred)

        return self

    def _build_tree(self, idx, g, h, depth):
        # Filter active gradients for this node
        g_k = np.zeros_like(g)
        h_k = np.zeros_like(h)
        g_k[idx] = g[idx]
        h_k[idx] = h[idx]

        if depth >= self.max_depth or idx.size < self.min_samples_leaf:
            w = -np.sum(g_k) / (self.gamma + np.sum(h_k) + 1e-12)
            return Node(is_leaf=True, depth=depth, weight=self.learning_rate * w)

        # --- SQUIRREL PROTOCOL: Secure Gradient Aggregation (BinMatVec) ---
        
        # Alice's local aggregation: M_0 * g^(k)
        host_G = self.host.M.dot(g_k)
        host_H = self.host.M.dot(h_k)
        
        client_G = np.array([]) 
        client_H = np.array([])
        
        if self.host.network:
            # A2H: Encrypt gradients to RLWE Ciphertexts (Simulated as dicts)
            g_k_enc = {"encrypted_data": g_k.tolist(), "pub_key_owner": self.host.name}
            h_k_enc = {"encrypted_data": h_k.tolist(), "pub_key_owner": self.host.name}
            
            # RPC: Remote BinMatVec
            client_G_enc = self.host.network.call_peer("bin_mat_vec", args={"encrypted_gradients": g_k_enc})
            client_H_enc = self.host.network.call_peer("bin_mat_vec", args={"encrypted_gradients": h_k_enc})
            
            # H2A: Decrypt back to arithmetic shares
            client_G = np.array(client_G_enc["encrypted_data"])
            client_H = np.array(client_H_enc["encrypted_data"])

        # --- SQUIRREL PROTOCOL: Secure Argmax (Simulated) ---
        best_gain = -1e18
        best = None
        
        def evaluate_gains(party_name, G_stats, H_stats):
            nonlocal best_gain, best
            n_bins = self.host.n_bins
            m_features = len(G_stats) // n_bins
            
            for j in range(m_features):
                G = G_stats[j*n_bins : (j+1)*n_bins]
                H = H_stats[j*n_bins : (j+1)*n_bins]
                
                G_cum = np.cumsum(G)
                H_cum = np.cumsum(H)
                G_tot = G_cum[-1] if len(G_cum) > 0 else 0
                H_tot = H_cum[-1] if len(H_cum) > 0 else 0
                
                for u in range(0, n_bins - 1):
                    GL, HL = G_cum[u], H_cum[u]
                    GR, HR = G_tot - GL, H_tot - HL
                    
                    gain = (GL * GL) / (self.gamma + HL + 1e-12) + \
                           (GR * GR) / (self.gamma + HR + 1e-12) - \
                           (G_tot * G_tot) / (self.gamma + H_tot + 1e-12)
                           
                    if np.isfinite(gain) and gain > best_gain:
                        best_gain = gain
                        best = (party_name, j, u)

        # Evaluate Alice's gains locally
        evaluate_gains(self.host.name, host_G, host_H)
        
        # Evaluate Bob's gains securely
        if len(client_G) > 0:
            evaluate_gains('bob', client_G, client_H)

        if best is None:
            w = -np.sum(g_k) / (self.gamma + np.sum(h_k) + 1e-12)
            return Node(is_leaf=True, depth=depth, weight=self.learning_rate * w)

        owner, feat_idx, bin_u = best

        # --- SQUIRREL PROTOCOL: Open Split Identifier ---
        if owner == self.host.name:
            edges = self.host.bin_edges[feat_idx]
            thr = float(np.max(edges)) if edges.size == 0 else float(edges[min(bin_u, edges.size - 1)])
            b_star = (self.host.X[:, feat_idx] <= thr).astype(bool)
        else:
            # Ask Bob to evaluate the split based on his winning (feat_idx, bin_u)
            b_star_list = self.host.network.call_peer("evaluate_split_indicator", args={
                "feat_idx": feat_idx, "bin_u": bin_u
            })
            b_star = np.array(b_star_list, dtype=bool)

        left_idx = idx[b_star[idx]]
        right_idx = idx[~b_star[idx]]

        if left_idx.size < self.min_samples_leaf or right_idx.size < self.min_samples_leaf:
            w = -np.sum(g_k) / (self.gamma + np.sum(h_k) + 1e-12)
            return Node(is_leaf=True, depth=depth, weight=self.learning_rate * w)

        # PRIVACY NOTE: We DO NOT store Bob's threshold. We just store the bin index.
        thr_store = thr if owner == self.host.name else float(bin_u)
        node = Node(is_leaf=False, depth=depth, owner=owner, feat_idx=feat_idx, thr=thr_store)
        
        node.left = self._build_tree(left_idx, g, h, depth+1)
        node.right = self._build_tree(right_idx, g, h, depth+1)
        return node

    def _update_predictions(self, tree, y_pred):
        stack = [(tree, np.arange(self.n, dtype=int))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf:
                y_pred[idx] += node.weight
            else:
                if node.owner == self.host.name:
                    b_star = (self.host.X[idx, node.feat_idx] <= node.thr)
                else:
                    b_star = np.array(self.host.network.call_peer("evaluate_split_indicator", args={
                        "feat_idx": node.feat_idx, "bin_u": int(node.thr), "idx": idx.tolist()
                    }), dtype=bool)
                    
                left_idx = idx[b_star]
                right_idx = idx[~b_star]
                
                if left_idx.size:
                    stack.append((node.left, left_idx))
                if right_idx.size:
                    stack.append((node.right, right_idx))

    def predict(self, threshold=0.5):
        logits = np.zeros(self.n, dtype=float)
        for tree in self.forest:
            stack = [(tree, np.arange(self.n, dtype=int))]
            while stack:
                node, idx = stack.pop()
                if node.is_leaf:
                    logits[idx] += node.weight
                else:
                    if node.owner == self.host.name:
                        b_star = (self.host.X[idx, node.feat_idx] <= node.thr)
                    else:
                        b_star = np.array(self.host.network.call_peer("evaluate_split_indicator", args={
                            "feat_idx": node.feat_idx, "bin_u": int(node.thr), "idx": idx.tolist()
                        }), dtype=bool)
                        
                    left_idx = idx[b_star]
                    right_idx = idx[~b_star]
                    if left_idx.size:
                        stack.append((node.left, left_idx))
                    if right_idx.size:
                        stack.append((node.right, right_idx))
        
        # Standard sigmoid for the final cleartext prediction
        p = 1.0 / (1.0 + np.exp(-logits))
        return (p >= threshold).astype(int)
