import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

clsStat = []
risks = np.array([[5, 8], [5, 7], [3, 2], [4, 10], [7, 10],
                 [10, 10], [8, 4], [7, 7], [6, 10], [9, 1]])
for i in range(1, len(risks)+1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(risks)
    clsStat.append(kmeans.inertia_)


plt.plot(range(1, len(risks)+1), clsStat)
plt.xlabel("Количество кластеров")
plt.ylabel("Разброс")
plt.show()
