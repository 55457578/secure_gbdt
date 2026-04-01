import pandas as pd
import numpy as np

def generate_complex_vfl_data(n_samples=5000, noise_level=0.05):
    np.random.seed(42)

    # --- ALICE'S FEATURES (Demographics & Employment) ---
    age = np.random.uniform(18, 80, n_samples)
    income = np.random.normal(60000, 25000, n_samples)
    income = np.clip(income, 20000, 200000)
    employment_years = np.random.uniform(0, 40, n_samples)
    
    # --- BOB'S FEATURES (Financial & Credit) ---
    credit_score = np.random.normal(650, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850)
    debt_to_income_ratio = np.random.uniform(0.0, 0.8, n_samples)
    recent_late_payments = np.random.poisson(1.5, n_samples) # heavily skewed

    # --- THE COMPLEX TARGET LOGIC (Hidden from both parties) ---
    # The true label is 1 (Default/Risk) if specific complex conditions are met across BOTH datasets.
    
    # Rule 1: Low credit score AND high debt (Requires Bob's data)
    rule_1 = (credit_score < 580) & (debt_to_income_ratio > 0.4)
    
    # Rule 2: Young, low income, but recent late payments (Requires Alice + Bob)
    rule_2 = (age < 25) & (income < 40000) & (recent_late_payments > 0)
    
    # Rule 3: High income but catastrophic debt and bad credit (Requires Alice + Bob)
    rule_3 = (income > 100000) & (debt_to_income_ratio > 0.6) & (credit_score < 650)
    
    # Base target
    target = (rule_1 | rule_2 | rule_3).astype(int)

    # Add random noise (flip a percentage of labels) to make it harder for the trees
    flip_mask = np.random.rand(n_samples) < noise_level
    target = np.where(flip_mask, 1 - target, target)

    # --- CREATE DATAFRAMES ---
    # Alice holds her features and the target 'y'
    alice_df = pd.DataFrame({
        'age': np.round(age, 1),
        'income': np.round(income, 2),
        'employment_years': np.round(employment_years, 1),
        'target': target
    })

    # Bob holds only his features
    bob_df = pd.DataFrame({
        'credit_score': np.round(credit_score, 0),
        'dti_ratio': np.round(debt_to_income_ratio, 3),
        'late_payments': recent_late_payments
    })

    # Save to CSV
    alice_df.to_csv('alice_complex.csv', index=False)
    bob_df.to_csv('bob_complex.csv', index=False)
    
    print(f"Generated {n_samples} rows of complex data.")
    print(f"Base Default Rate (Target=1): {target.mean() * 100:.2f}%")

if __name__ == "__main__":
    generate_complex_vfl_data(n_samples=5000, noise_level=0.10)