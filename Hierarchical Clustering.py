#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing required libraries
import pandas as pb
import matplotlib.pyplot as plt

#Reading dataset
dataset = pb.read_csv("Mall_Customers-211105-191711.csv")
x=dataset.iloc[:,:].values


# In[4]:


x


# In[9]:


#model
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method="ward"))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")


# In[6]:


#training model
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters = 5).fit_predict(x)
clustering


# In[10]:


#visualising model
plt.scatter(x[clustering == 0 , 0],x[clustering == 0 , 1],c= "Blue" , label="cluster 1")
plt.scatter(x[clustering == 1 , 0],x[clustering == 1 , 1],c= "green" , label="cluster 2")
plt.scatter(x[clustering == 2 , 0],x[clustering == 2 , 1],c= "red" , label="cluster 3")
plt.scatter(x[clustering == 3 , 0],x[clustering == 3 , 1],c= "orange" , label="cluster 4")
plt.scatter(x[clustering == 4 , 0],x[clustering == 4 , 1],c= "violet" , label="cluster 5")
plt.title("Cluster of customer")
plt.xlabel("Annual Income(k$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()


# In[ ]:




