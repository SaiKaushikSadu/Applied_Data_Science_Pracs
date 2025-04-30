# Outlier Detection
# DBSCAN / LOF - Density based
# KNN - Distance based
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')
df1 = df
df1.drop('Id', axis=1, inplace=True)
df.boxplot()
plt.show()

X = df[['SepalLengthCm', 'SepalWidthCm']]

# K Nearest Neighbors
knn = NearestNeighbors(n_neighbors=5)
knn.fit(X)
distances, _ = knn.kneighbors(X)
avg_distances = distances.mean(axis=1)
threshold = np.percentile(avg_distances, 95)
outliers = avg_distances > threshold

plt.scatter(X['SepalLengthCm'], X['SepalWidthCm'], c=outliers)
plt.show()

# DBSCAN
db = DBSCAN(eps=0.4, min_samples=10).fit(X)
colors = db.labels_
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], c=colors)
plt.show()

outliers_db = df[db.labels_ == -1]
print(outliers_db)

df.head().to_html('iris7.html')