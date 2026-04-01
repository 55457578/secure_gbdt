import pandas as pd
import numpy as np

# Generate 500 rows of aligned data
np.random.seed(42)

# Alice has 3 features and the target (y)
alice_df = pd.DataFrame({
    'age': np.random.randint(18, 80, 500),
    'income': np.random.randint(30000, 150000, 500),
    'credit_score': np.random.randint(300, 850, 500),
    'target': np.random.randint(0, 2, 500) # The label 'y'
})
alice_df.to_csv('alice_data.csv', index=False)

# Bob has 3 different features for the exact same 500 people
bob_df = pd.DataFrame({
    'recent_purchases': np.random.randint(0, 50, 500),
    'website_visits': np.random.randint(0, 100, 500),
    'support_tickets': np.random.randint(0, 10, 500)
})
bob_df.to_csv('bob_data.csv', index=False)