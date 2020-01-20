#!/usr/bin/env python
# coding: utf-8

# ## EM algorithm and implementation

# ### b

# In[56]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from collections import defaultdict
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[57]:


x = []
y = []
with open("data.txt", "r") as file:
    for line in file.readlines()[1:]:
        sp = line.split()
        x.append(float(sp[1]))
        y.append(int(sp[2]))


# In[58]:


data = []
for i in range(len(x)):
    data.append([x[i],y[i]])


# In[59]:


plt.scatter(x, y, alpha=0.5)
plt.show()


# -----

# ### c

# In[60]:


means_x1 = []
means_y1 = []
means_x2 = []
means_y2 = []

gmm = mixture.GaussianMixture(n_components=2,covariance_type='spherical', init_params="random", warm_start = True, max_iter= 1)

boolean = False
while boolean is False:
    gmm.fit(data)
    means_x1.append(gmm.means_[0][0])
    means_y1.append(gmm.means_[0][1])
    means_x2.append(gmm.means_[1][0])
    means_y2.append(gmm.means_[1][1])
    
    boolean = gmm.converged_
    


# In[65]:


colors = ["red"]*11 + ["blue"]*11
means_x = means_x1 + means_x2
means_y = means_y1 + means_y2


# In[66]:


plt.scatter(means_x, means_y, c=colors, alpha=1)
plt.scatter(x, y, c= "green", alpha=0.5)
plt.show()


# In[67]:


means_final = []
iterations = defaultdict(int)

for i in range(50):
    weights = np.random.rand(2)
    means = np.random.rand(2,2)
    precisions = np.random.rand(2)
    gmm = mixture.GaussianMixture(n_components=2,covariance_type='spherical',
                                 weights_init=None, means_init=means, precisions_init=None)
    gmm.fit(data)
    means_final.append(gmm.means_)
    iterations[gmm.n_iter_] += 1


# In[68]:


pos = np.arange(len(iterations.keys()))
width = 1.0 

plt.bar(iterations.keys(), iterations.values(), width)


# ____

# ### d

# In[69]:


kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_


# In[70]:


group0 = []
group1 = []

for i in range(len(data)):
    if labels[i] == 1:
        group0.append(data[i])
    else:
        group1.append(data[i])


# In[71]:


x0 = []
y0 = []
x1 = []
y1 = []

for i in group0:
    x0.append(i[0])
    y0.append(i[1])

for i in group1:
    x1.append(i[0])
    y1.append(i[1])
    


# In[72]:


plt.scatter(x0, y0, c="red", alpha=0.5)
plt.scatter(x1, y1, c="blue", alpha=0.5)
plt.scatter(centers[0][0], centers[0][1], c="yellow", s=100, alpha=1)
plt.scatter(centers[1][0], centers[1][1], c="yellow", s=100, alpha=1)

plt.show()


# In[73]:


import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel


# In[74]:


gmm_kmeansinit = mixture.GaussianMixture(n_components=2,covariance_type='spherical', init_params="kmeans")


# In[75]:


gmm_kmeansinit.fit(data)


# In[76]:


gmm_kmeansinit.n_iter_


# In[80]:


means_x1_k = []
means_y1_k = []
means_x2_k = []
means_y2_k = []

gmm_kmeansinit = mixture.GaussianMixture(n_components=2,covariance_type='spherical', 
                                         init_params="kmeans", warm_start = True, max_iter= 1)

boolean = False
while boolean is False:
    gmm_kmeansinit.fit(data)
    means_x1_k.append(gmm_kmeansinit.means_[0][0])
    means_y1_k.append(gmm_kmeansinit.means_[0][1])
    means_x2_k.append(gmm_kmeansinit.means_[1][0])
    means_y2_k.append(gmm_kmeansinit.means_[1][1])
    
    boolean = gmm_kmeansinit.converged_
    


# In[83]:


colors = ["red"]*2 + ["blue"]*2
means_x_k = means_x1_k + means_x2_k
means_y_k = means_y1_k + means_y2_k


# In[86]:


means_x1_k


# In[84]:


plt.scatter(means_x_k, means_y_k, c=colors, alpha=1)
plt.scatter(x, y, c= "green", alpha=0.5)
plt.show()


# In[89]:


means_final = []
iterations = defaultdict(int)

for i in range(50):
    weights = np.random.rand(2)
    means = np.random.rand(2,2)
    precisions = np.random.rand(2)
    gmm_kmeansinit = mixture.GaussianMixture(n_components=2,covariance_type='spherical',
                                 init_params= "kmeans", precisions_init=None)

    gmm_kmeansinit.fit(data)
    means_final.append(gmm_kmeansinit.means_)
    iterations[gmm_kmeansinit.n_iter_] += 1


# In[90]:


pos = np.arange(len(iterations.keys()))
width = 1.0 

plt.bar(iterations.keys(), iterations.values(), width)


# In[92]:


iterations

