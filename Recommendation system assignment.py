#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation


# In[2]:


book= pd.read_csv('F:/Dataset/book2.csv')


# In[4]:


book


# In[5]:


book2= book.iloc[:,1:]


# In[6]:


book2


# In[7]:


book2.sort_values(['User.ID'])


# In[8]:


len(book2['User.ID'].unique())


# In[10]:


len(book2['Book.Title'].unique())


# In[11]:


book3= book2.pivot_table(index='User.ID',columns="Book.Title",values='Book.Rating').reset_index(drop=True)


# In[12]:


book3


# In[13]:


book3.index=book2['User.ID'].unique()


# In[14]:


book3


# In[15]:


book3.fillna(0,inplace=True)


# In[16]:


book3


# In[23]:


usersim= 1-pairwise_distances(book3.values,metric='cosine')


# In[24]:


usersim


# In[26]:


usersim2=pd.DataFrame(usersim)


# In[27]:


usersim2


# In[33]:


usersim2.index=book2['User.ID'].unique()


# In[34]:


usersim2.columns=book2['User.ID'].unique()


# In[35]:


usersim2


# In[36]:


np.fill_diagonal(usersim,0)


# In[38]:


usersim


# In[39]:


usersim2.idxmax(axis=1)


# In[45]:


book2[(book2['User.ID']==162107) | (book2['User.ID']==276726)]


# In[47]:


book2[(book2['User.ID']==276729) | (book2['User.ID']==276726)]


# In[48]:


user1=book2[(book2['User.ID']==276729)]
user2=book2[(book2['User.ID']==276726)]


# In[49]:


user1['Book.Title']


# In[50]:


user2['Book.Title']


# In[51]:


pd.merge(user1,user2,on='Book.Title',how='outer')


# In[ ]:




