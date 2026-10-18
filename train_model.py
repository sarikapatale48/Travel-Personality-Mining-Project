import os
import joblib
import matplotlib.pyplot as plt

from src.data_preprocessing import load_data, preprocess_data, scale_data
from src.clustering import find_best_k, train_kmeans


DATA_PATH = "dataset/travel_review.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# --------------------------------
# 1. Load dataset
# --------------------------------

df = load_data(DATA_PATH)

print("Original dataset shape:", df.shape)


# --------------------------------
# 2. Preprocess
# --------------------------------

df = preprocess_data(df)

print("After preprocessing:", df.shape)


# --------------------------------
# 3. Scale
# --------------------------------

scaled_data, scaler = scale_data(df)


# --------------------------------
# 4. Find best K
# --------------------------------

results = find_best_k(scaled_data)

print("\nSilhouette Scores:")

for k, score in results:
    print(f"K = {k}, Score = {score:.4f}")


# --------------------------------
# 5. Plot silhouette scores
# --------------------------------

k_values = [x[0] for x in results]
scores = [x[1] for x in results]

plt.figure(figsize=(8, 5))

plt.plot(k_values, scores, marker="o")

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs Number of Clusters")

plt.grid(True)

plt.savefig("outputs/silhouette_score.png")

plt.show()


# --------------------------------
# 6. Train final model
# --------------------------------

best_k = max(results, key=lambda x: x[1])[0]

print("\nSelected K:", best_k)

model, labels = train_kmeans(
    scaled_data,
    best_k
)


# --------------------------------
# 7. Save models
# --------------------------------

joblib.dump(model, "models/kmeans_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")


# --------------------------------
# 8. Save cluster labels
# --------------------------------

df["Cluster"] = labels

df.to_csv(
    "outputs/clustered_data.csv",
    index=False
)

print("\nModel training completed!")

print("Files saved inside models/ and outputs/")
