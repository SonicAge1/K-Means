from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

iris =datasets.load_iris()
irisFeatures = iris["data"]
irisFeaturesName = iris['feature_names']
irisLabels = iris['target']


model = KMeans(n_clusters=4, random_state=1).fit(irisFeatures)


# 获取聚类中心和聚类标签
centroids = model.cluster_centers_
labels = model.labels_

# 可视化结果
plt.figure(figsize=(8, 6))
plt.scatter(irisFeatures[:, 0], irisFeatures[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', alpha=0.5, marker='X')
plt.title('K-means Clustering')
plt.xlabel(f'{irisFeaturesName[0]}')
plt.ylabel(f'{irisFeaturesName[1]}')
plt.show()