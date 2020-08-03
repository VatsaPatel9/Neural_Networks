# 
#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import random


# In[14]:


#lsit of 1st cluster 
x1=[] 
y1=[]
z1=[]
for i in range (0,50):
    x1.append(float(random.uniform(-1,1)))
    y1.append(float(random.uniform(-1,1)))
    z1.append(float(random.uniform(-1,1)))
cluster1=len(x1)    


# In[11]:


#lsit of 2nd cluster 
# x2=[] 
# y2=[]
# z2=[]
for i in range (0,50):
    x1.append(float(random.uniform(-1,1)))
    y1.append(float(random.uniform(-1,1)))
    z1.append(float(random.uniform(3,4)))
    


# In[12]:


X=float(input("enter x"))
Y=float(input("enter y"))
Z=float(input("enter z"))
point1=(X,Y,Z)
print(point1)




op=[]
for i in range (0,len(x1)):
    D=[]
    point2=(x1[i],y1[i],z1[i])
    d=np.sqrt(((point1[0] - point2[0])**2)+((point1[1]-point2[1])**2)+(point1[2]-point2[2])**2)
    print(d)
    if i<cluster1:
        D.append(0)
    else:
        D.append(1)
    D.append(d)
    op.append(D)
    
k=int(input("enter k"))

# D.sort()
# print(D)
sorted(op,key=lambda l:l[1])

c1=0
c2=0

for i in range(0,k):
    print(op[i])

    

    if op[2*i]==0:
        c1=c1+1
    else:
        c2=c2+1
count=max(c1,c2)
accuracy=(count/k)*100
print(accuracy)
    


# In[ ]:





# In[56]:





# In[ ]:




