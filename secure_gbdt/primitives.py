# secure_gbdt/primitives.py
import numpy as np

# ==========================================
# CRYPTOGRAPHIC DATA STRUCTURES (Simulated)
# ==========================================

class SecretShare:
    """Represents a 2-out-of-2 additive secret share of a value."""
    def __init__(self, data, party_id):
        self.data = np.asarray(data, dtype=float)
        self.party_id = party_id

class RLWECiphertext:
    """Represents a Ring Learning with Errors Homomorphic Encryption."""
    def __init__(self, encrypted_data, pub_key_owner):
        self.encrypted_data = encrypted_data
        self.pub_key_owner = pub_key_owner

# ==========================================
# SMPC IDEAL FUNCTIONALITIES (Simulated)
# ==========================================
# In a real system, these trigger network calls (OT, Garbled Circuits, etc.)

def F_mul(share0: SecretShare, share1: SecretShare) -> np.ndarray:
    """Secure Multiplication: <z> = <x> * <y>"""
    # SIMULATION: Reconstruct, multiply, and return the true value
    # Real implementation requires Beaver Triples or OT
    return share0.data * share1.data

def F_greater(share_x: SecretShare, threshold: float) -> np.ndarray:
    """Secure Comparison: returns boolean share of x > threshold"""
    # SIMULATION
    return (share_x.data > threshold).astype(float)

def F_cor(choice_bit_share, correlation_func_data) -> np.ndarray:
    """Correlated Oblivious Transfer (COT) for invariant updates"""
    # SIMULATION: Used to compute b_star * gradients securely
    return choice_bit_share * correlation_func_data

# ==========================================
# SQUIRREL PROTOCOLS
# ==========================================

def seg3_sigmoid(share_x: SecretShare) -> np.ndarray:
    """
    Squirrel's Seg3Sigmoid Protocol (Figure 5).
    Corrected segment masking to prevent division-by-zero gradients.
    """
    tau = 5.6
    
    # F_greater returns 1 if x > threshold
    is_gt_neg_tau = F_greater(share_x, -tau) # 1 if x > -5.6
    is_gt_pos_tau = F_greater(share_x, tau)  # 1 if x > 5.6
    
    # Correct segment masking:
    b_left = 1.0 - is_gt_neg_tau               # 1 if x <= -5.6
    b_right = is_gt_pos_tau                    # 1 if x > 5.6
    b_mid = is_gt_neg_tau - is_gt_pos_tau      # 1 if -5.6 < x <= 5.6
    
    # Middle segment
    mid_val = np.clip(share_x.data, -tau, tau)
    
    # Using precise sigmoid for the mock to ensure perfect tree splitting
    f_x = 1.0 / (1.0 + np.exp(-mid_val))
    
    return (b_left * 0.0) + (b_mid * f_x) + (b_right * 1.0)

def A2H(share: SecretShare, target_party) -> RLWECiphertext:
    """Arithmetic Share to Homomorphic Encryption (A2H)"""
    # SIMULATION: Wrap the data in a mock ciphertext object
    return RLWECiphertext(share.data, target_party)

def H2A(ciphertext: RLWECiphertext, target_party) -> SecretShare:
    """Homomorphic Encryption to Arithmetic Share (H2A)"""
    # SIMULATION: Unwrap the ciphertext into a secret share
    return SecretShare(ciphertext.encrypted_data, target_party)
