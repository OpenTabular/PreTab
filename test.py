from pretab.preprocessor import Preprocessor
import numpy as np
import pandas as pd

# Simulated dataset
np.random.seed(42)
n_samples = 100

df = pd.DataFrame(
    {
        "age": np.random.randint(18, 80, size=n_samples),
        "income": np.random.normal(50000, 15000, size=n_samples),
        "iq": np.random.normal(0, 150, size=n_samples),
        "debt": np.random.normal(50000, 15000, size=n_samples),
        "city": np.random.choice(["Berlin", "Paris", "London"], size=n_samples),
        "gender": np.random.choice(["M", "F"], size=n_samples),
    }
)

y = df["income"] * 0.3 + df["age"] * 12 + np.random.normal(0, 1000, size=n_samples)

# Define per-feature preprocessing strategy
feature_preprocessing = {
    "age": "standardization",
    "income": "rbf",
    "city": "one-hot",
    "gender": "int",
}

# Instantiate the preprocessor
prepro = Preprocessor(
    feature_preprocessing=feature_preprocessing,
    n_knots=10,
    scaling_strategy="minmax",
    numerical_preprocessing="ple",  # default fallback
    categorical_preprocessing="int",  # default fallback
)

# Fit and transform
Xt = prepro.fit_transform(df, y, embeddings=np.random.randn(n_samples, 10))

# Check feature info
numerical_info, categorical_info, embedding_info = prepro.get_feature_info(verbose=True)

# Output a few checks
print("\nTransformed shape:", Xt.shape)
print("Numerical info:", numerical_info)
print("Categorical info:", categorical_info)
