import pandas as pd
import joblib

# Load clustered dataset
df = pd.read_csv("outputs/clustered_data.csv")

# Load trained model
model = joblib.load("models/kmeans_model.pkl")

# Find cluster averages
profile = df.groupby("Cluster").mean(numeric_only=True)

print("\n==============================")
print("       CLUSTER PROFILES")
print("==============================")

print(profile.round(2))

# Show top 5 features for every cluster
for cluster in profile.index:

    print("\n==============================")
    print("Cluster:", cluster)
    print("==============================")

    top_features = (
        profile.loc[cluster]
        .sort_values(ascending=False)
        .head(5)
    )

    print(top_features.round(2))