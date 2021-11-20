# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.cluster import KMeans

# k = 6
# colors = ['red', 'green', 'black', 'yellow', 'magenta', "brown"]
# risks = np.array([[5, 8], [5, 7], [3, 2], [4, 10], [7, 10],
#                  [10, 10], [8, 4], [7, 7], [6, 10], [9, 1]])


# kmeans = KMeans(n_clusters=k, random_state=42)
# y = kmeans.fit_predict(risks)

# for i in range(k):
#     plt.scatter(risks[y == i, 0], risks[y == i, 1],
#                 s=100, c=colors[i], label=f"cluster{i}")
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
#             :, 1], s=150, c="blue", label="centers")
# plt.xlabel("Степеь реализации угрозы")
# plt.ylabel("Степень влияния угрозы на актив")
# plt.show()

def my_max(arr):
    mx = 0
    for i in arr:
        if i > mx:
            mx = i
    return mx


print(list(map(my_max, [[1,2,3], [4,5,6], [7,8,9]])))