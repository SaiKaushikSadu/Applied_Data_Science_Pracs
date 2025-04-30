# Unsupervised - KMeans + Metrics

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

df = pd.read_csv('./Iris.csv')

le = LabelEncoder()
df['Target'] = le.fit_transform(df['Species'])

kmeans = KMeans(n_clusters=3)
kmeans.fit(df[['SepalLengthCm', 'Target']])

df['k_means_label'] = kmeans.labels_

plt.scatter(x=df['SepalLengthCm'], y=df['Target'], c= df['k_means_label'])
plt.show()

ss = silhouette_score(df[['SepalLengthCm', 'Target']], df['k_means_label'], metric='euclidean')
print(ss)

ars = adjusted_rand_score(df['Target'], df['k_means_label'])
print(ars)

nms = normalized_mutual_info_score(df['Target'], df['k_means_label'])
print(nms)

df.head(10).to_html('iris1.html')