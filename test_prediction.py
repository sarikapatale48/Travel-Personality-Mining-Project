import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("dataset/travel_review.csv")

# Remove ID
df = df.drop(columns=[df.columns[0]])

# Convert to numeric
df = df.apply(pd.to_numeric, errors="coerce")

df = df.dropna()

# Load model and scaler
model = joblib.load("models/kmeans_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Test 1 - all low
test1 = pd.DataFrame(
    [[1] * len(df.columns)],
    columns=df.columns
)

# Test 2 - all high
test2 = pd.DataFrame(
    [[5] * len(df.columns)],
    columns=df.columns
)

# Test 3 - nature oriented
test3 = pd.DataFrame(
    [[3] * len(df.columns)],
    columns=df.columns
)

# Predictions
for name, test in [
    ("All Low", test1),
    ("All High", test2),
    ("Balanced", test3)
]:

    scaled = scaler.transform(test)

    prediction = model.predict(scaled)[0]

    distances = model.transform(scaled)[0]

    print("\n", name)

    print("Predicted Cluster:", prediction)

    print("Distances:")

    for i, distance in enumerate(distances):
        print(
            f"Cluster {i}: {distance:.2f}"
        )