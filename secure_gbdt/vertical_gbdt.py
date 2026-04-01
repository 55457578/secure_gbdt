# secure_gbdt/vertical_gbdt.py
import numpy as np
from .nodes import Node
from .primitives import sigmoid

class VerticalGBDT:
    def __init__(self, host_party, y, max_depth=4, n_trees=10,
                 min_samples_leaf=10, gamma=1e-3, learning_rate=0.1, epsilon=0.5):
        """
        host_party: The local Party object (Alice). Should have an active network adapter to reach Bob.
        y: labels in {0,1}, shape (n,)
        """
        self.host = host_party
        self.y = np.asarray(y, dtype=float)
        self.n = self.y.shape[0]
        self.max_depth = max_depth
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.forest = []

    def fit(self):
        # 1. Local binning
        self.host.fit_bins()
        
        # 2. Tell Bob to trigger his binning over RPC
        if self.host.network:
            print("[Network] Instructing Client to bin features...")
            self.host.network.call_peer("fit_bins", args={})
            
        y_pred = np.zeros(self.n, dtype=float)

        for tree_i in range(self.n_trees):
            print(f"Building Tree {tree_i + 1}/{self.n_trees}")
            p = sigmoid(y_pred)
            g = p - self.y
            h = p * (1.0 - p)

            idx_all = np.arange(self.n, dtype=int)
            tree = self._build_tree(idx_all, g, h, depth=0)
            self.forest.append(tree)
            self._update_predictions(tree, y_pred)

        return self

    def _build_tree(self, idx, g, h, depth):
        if depth >= self.max_depth or idx.size < self.min_samples_leaf:
            w = -g[idx].sum() / (self.gamma + h[idx].sum() + 1e-12)
            return Node(is_leaf=True, depth=depth, weight=self.learning_rate * w)

        # Evaluate Alice's best split locally
        host_best = self.host.get_best_split(
            idx.tolist(), g[idx].tolist(), h[idx].tolist(), 
            self.gamma, self.epsilon
        )

        # Evaluate Bob's best split over RPC
        client_best = {"gain": -1e18}
        if self.host.network:
            client_best = self.host.network.call_peer(
                "get_best_split",
                args={
                    "idx": idx.tolist(),
                    "g": g[idx].tolist(),
                    "h": h[idx].tolist(),
                    "gamma": self.gamma,
                    "epsilon": self.epsilon
                }
            )

        # Compare gains
        if host_best["gain"] == -1e18 and client_best["gain"] == -1e18:
            w = -g[idx].sum() / (self.gamma + h[idx].sum() + 1e-12)
            return Node(is_leaf=True, depth=depth, weight=self.learning_rate * w)

        if host_best["gain"] >= client_best["gain"]:
            owner = self.host.name
            feat_idx = host_best["feat_idx"]
            thr = host_best["thr"]
            # Get mask locally
            sel = np.array(self.host.evaluate_split(idx.tolist(), feat_idx, thr), dtype=bool)
        else:
            owner = "bob"
            feat_idx = client_best["feat_idx"]
            thr = client_best["thr"]
            # Get mask from Bob via RPC
            sel = np.array(
                self.host.network.call_peer(
                    "evaluate_split", 
                    args={"idx": idx.tolist(), "feat_idx": feat_idx, "thr": thr}
                ), dtype=bool
            )

        left_idx = idx[sel]
        right_idx = idx[~sel]

        if left_idx.size < self.min_samples_leaf or right_idx.size < self.min_samples_leaf:
            w = -g[idx].sum() / (self.gamma + h[idx].sum() + 1e-12)
            return Node(is_leaf=True, depth=depth, weight=self.learning_rate * w)

        node = Node(is_leaf=False, depth=depth, owner=owner, feat_idx=feat_idx, thr=thr)
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
                    sel = np.array(self.host.evaluate_split(idx.tolist(), node.feat_idx, node.thr), dtype=bool)
                else:
                    sel = np.array(
                        self.host.network.call_peer(
                            "evaluate_split", 
                            args={"idx": idx.tolist(), "feat_idx": node.feat_idx, "thr": node.thr}
                        ), dtype=bool
                    )
                    
                left_idx = idx[sel]
                right_idx = idx[~sel]
                
                if left_idx.size:
                    stack.append((node.left, left_idx))
                if right_idx.size:
                    stack.append((node.right, right_idx))

    def predict_proba(self):
        """
        Predicts based on the data currently loaded in the Parties.
        Assumes data alignment (Row X on Alice is Row X on Bob).
        """
        logits = np.zeros(self.n, dtype=float)
        for tree in self.forest:
            logits += self._tree_predict_logits(tree)
        return sigmoid(logits)

    def predict(self, threshold=0.5):
        return (self.predict_proba() >= threshold).astype(int)

    def _tree_predict_logits(self, tree):
        out = np.zeros(self.n, dtype=float)
        stack = [(tree, np.arange(self.n, dtype=int))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf:
                out[idx] += node.weight
            else:
                if node.owner == self.host.name:
                    sel = np.array(self.host.evaluate_split(idx.tolist(), node.feat_idx, node.thr), dtype=bool)
                else:
                    sel = np.array(
                        self.host.network.call_peer(
                            "evaluate_split", 
                            args={"idx": idx.tolist(), "feat_idx": node.feat_idx, "thr": node.thr}
                        ), dtype=bool
                    )
                left_idx = idx[sel]
                right_idx = idx[~sel]
                if left_idx.size:
                    stack.append((node.left, left_idx))
                if right_idx.size:
                    stack.append((node.right, right_idx))
        return out