import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)


inertia_list = []
for k in np.arange(1,10):
    kmn = KMeans(n_clusters=k)
    kmn.fit(df.values)
    inertia_list.append(kmn.inertia_)

plt.plot(np.arange(1,10),inertia_list,'ro-')
plt.xlabel('number of clusters')
plt.ylabel('overlap')
plt.show()

kmn = KMeans(n_clusters=3)
kmn.fit(df.values)
labels = kmn.predict(df.values)

xs = df.values[:,1]
ys = df.values[:,0]
plt.scatter(ys,xs,c=labels)
plt.legend(labels,loc='best')

columns = ['Cluster']
df_cluster = pd.DataFrame(labels,index=None, columns=columns)
F_df = df.join(df_cluster)

