# secure_gbdt/nodes.py
from dataclasses import dataclass

@dataclass
class Node:
    is_leaf: bool
    depth: int
    owner: str = None       # 'alice' or 'bob' when not leaf
    feat_idx: int = None    # feature index local to owner
    thr: float = None       # threshold value
    left: "Node" = None
    right: "Node" = None
    weight: float = None    # leaf weight (logit contribution)
