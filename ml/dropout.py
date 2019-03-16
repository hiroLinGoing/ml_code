#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[13]:


#X输入 drop_probability丢失的概率
def dropout(X, drop_probability):  
    keep_probability = 1 - drop_probability
    assert 0 <= drop_probability <= 1
    if keep_probability == 0:
        return X.zeros_like()
    sample=np.random.binomial(1,keep_probability,size=X.shape) 
    X *=sample
    #这里除是因为要使得输入矩阵期望保持一致
    scale = 1/keep_probability 
    return X*scale


# In[22]:


A = np.arange(20).reshape((5, 4))
dropout(A, 0.5)

