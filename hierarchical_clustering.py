import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage
)

from sklearn.preprocessing import StandardScaler


# Load data

df = pd.read_csv(
    "dataset/travel_review.csv"
)


# Remove ID

df = df.drop(
    columns=[df.columns[0]]
)


# Convert to numeric

df = df.apply(
    pd.to_numeric,
    errors="coerce"
)


df = df.dropna()


# Scale

scaler = StandardScaler()

data = scaler.fit_transform(df)


# Use a sample for dendrogram
# because plotting all 5456 records
# makes the graph unreadable

sample = data[:500]


# Hierarchical clustering

linked = linkage(
    sample,
    method="ward"
)


# Dendrogram

plt.figure(
    figsize=(14, 7)
)

dendrogram(
    linked,
    truncate_mode="level",
    p=5
)

plt.title(
    "Hierarchical Clustering Dendrogram"
)

plt.xlabel("Tourists")

plt.ylabel("Distance")

plt.tight_layout()

plt.savefig(
    "outputs/dendrogram.png"
)

plt.show()