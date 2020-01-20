#!/usr/bin/env python
# coding: utf-8

# ## Multidimensional scaling for genetic population differences

# In[77]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from pyclustering.cluster.kmedoids import kmedoids


# In[78]:


data = np.load("mds-population.npz")
print (data['D']) # Distance matrix
print (data['population_list']) # List of populations


# In[79]:


data_array = []

for i in range(len(data["D"])):
    data_array.append(list(data["D"][i]))
    
data_array = np.array(data_array) 


# In[80]:


emmbeded = MDS(n_components=2, dissimilarity='precomputed')
X_transformed = emmbeded.fit_transform(data_array)


# In[84]:


values = [2,4,6,8,10,12,14,16,18,20]

for i in range(2,42):
    emmbeded = MDS(n_components=i, dissimilarity='precomputed')
    X_transformed_a = emmbeded.fit_transform(data_array)
    u, s, vh = np.linalg.svd(X_transformed_a, full_matrices=True)
    print(s)
    nonsing  = np.sum(s < 0.1)
    print("m: {}, nonsingular values: {}".format(i, nonsing))
    print("--------------------------------------")


# In[86]:


m


# In[20]:


x = []
y = []

for i in X_transformed:
    x.append(i[0])
    y.append(i[1]) 


# In[21]:


labels = []
for i in data['population_list']:
    i = str(i)
    labels.append(i.lstrip("b'").strip("''")) 


# In[22]:


fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(x, y )

for i, labels in enumerate(labels):
    ax.annotate(labels, (x[i], y[i]))
    
plt.show()


# -------

# ## K-means

# In[14]:


def kMeans():
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    numClusters = list(range(1,21))
    sse = [0] * len(numClusters)

    for k in numClusters:
        #print(k)
        kmeans = KMeans(k, n_init=10, init='random').fit(X_transformed)
        sse[k-1] += kmeans.inertia_
    
    plt.plot(list(range(1,21)), sse, marker='x')
    plt.title('SSE for different Ks')
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.grid()
    plt.savefig("SSE for Ks")


# In[15]:


kMeans()


# In[192]:


kmeans = KMeans(5, n_init=10, init='random').fit(X_transformed)


# In[193]:


kmeans.labels_


# In[223]:


from collections import defaultdict
d = defaultdict(list)

for i in range(len(kmeans.labels_)):
    d[kmeans.labels_[i]] += [X_transformed[i]]


# In[237]:


colors = ["r", "b", "g", "y", "orange"]
for i in range(5):
    plt.scatter([dx[0] for dx in d[i]], [dx[1] for dx in d[i]], c=colors[i], alpha=0.5)

plt.show()


# ------

# In[17]:


from scipy.cluster import hierarchy


# In[28]:


import scipy.spatial.distance as ssd
distArray = ssd.squareform(data_array)


# In[38]:


Z = hierarchy.linkage(distArray, "average")
plt.figure(figsize=(20,10))
dn = hierarchy.dendrogram(Z, labels = labels)


# In[47]:


from scipy.cluster.hierarchy import ward, fcluster
cut = fcluster(Z, t=150, criterion='distance')


# In[48]:


cut


# In[50]:


from collections import defaultdict
d_cut = defaultdict(list)

for i in range(len(cut)):
    d_cut[cut[i]] += [X_transformed[i]]


# In[51]:


colors = ["r", "b", "g", "y", "orange"]
for i in range(5):
    plt.scatter([dx[0] for dx in d_cut[i]], [dx[1] for dx in d_cut[i]], c=colors[i], alpha=0.5)

plt.show()


# ______

# In[29]:


kmedoids_instance = kmedoids(data_array, [5, 4, 2, 1, 3])
kmedoids_instance.process()

clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()


# In[38]:


clusters_all = []

for i in clusters:
    for j in i:
        clusters_all.append(list(X_transformed[j]))


# In[44]:


xx = []
yy = []
for i in range(len(clusters_all)):
    xx.append(clusters_all[i][0])
    yy.append(clusters_all[i][1])


# In[70]:


color = ["y"]*len(clusters[0])+ ["r"]*len(clusters[1])+ ["b"]*len(clusters[2])+ ["orange"]*len(clusters[3])+ ["g"]*len(clusters[4])


# In[72]:


labels = []
for i in data['population_list']:
    i = str(i)
    labels.append(i.lstrip("b'").strip("''")) 


# In[73]:


fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(xx, yy, c=color)

for i, labels in enumerate(labels):
    ax.annotate(labels, (xx[i], yy[i]))
    
plt.show()

