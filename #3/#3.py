#!/usr/bin/env python
# coding: utf-8

# In[113]:


import numpy as np
import random


# In[114]:


# learning rate

lr=0.001

#  input x1
# x1 = [0,0,1,1,1,4,0,4]

# #  input x2
# x2 = [0,1,0,1,1,0,4,4]

# # input x3
# x3 = [0,1,1,1,0,4,4,4]

# #  expected output
# op = [1,1,1,1,1,0,0,0]

x1=[] 
x2=[]
x3=[]
for i in range (0,50):
    x1.append(float(random.uniform(-1,1)))
    x2.append(float(random.uniform(-1,1)))
    x3.append(float(random.uniform(-1,1)))
    op.append(0)

for i in range (0,50):
    x1.append(float(random.uniform(-1,1)))
    x2.append(float(random.uniform(-1,1)))
    x3.append(float(random.uniform(3,4)))
    op.append(1)



# In[115]:


# Generating random weights 
#y1
w1 = random.uniform(-2.0,2)
w2 = random.uniform(-2.0,2)
w3 = random.uniform(-2.0,2)
w10 = random.uniform(-2.0,2) #bias
#y2
w4 = random.uniform(-2.0,2)
w5 = random.uniform(-2.0,2)
w6 = random.uniform(-2.0,2)
w11 = random.uniform(-2.0,2) #bias
#y3
w7 = random.uniform(-2.0,2)
w8 = random.uniform(-2.0,2)
w9 = random.uniform(-2.0,2)
w12 = random.uniform(-2.0,2) #bias

#z1
w11 = random.uniform(-2.0,2)
w21 = random.uniform(-2.0,2)
w31= random.uniform(-2.0,2)
w01= random.uniform(-2.0,2) #bias
#z2
w12 = random.uniform(-2.0,2)
w22 = random.uniform(-2.0,2)
w32 = random.uniform(-2.0,2)
w02 = random.uniform(-2.0,2)#bias
#z3
w13 = random.uniform(-2.0,2)
w23 = random.uniform(-2.0,2)
w33 = random.uniform(-2.0,2)
w03 = random.uniform(-2.0,2)#bias
#z4
w14 = random.uniform(-2.0,2)
w24 = random.uniform(-2.0,2)
w34 = random.uniform(-2.0,2)
w04 = random.uniform(-2.0,2)#bias

#op
w111 = random.uniform(-2.0,2)
w222 = random.uniform(-2.0,2)
w333 = random.uniform(-2.0,2)
w444 = random.uniform(-2.0,2)
w000 = random.uniform(-2.0,2)

print("w1",w1)
print("w2",w2)
print("w3",w3)
print("w10",w10)


# In[116]:


#forward checking

count=0
while(count<1000):
         
    for i in range(0,len(x1)):
        
        #layer 1
        y1 = x1[i]*w1 + x2[i]*w2 + x3[i]*w3 + w10 
        y2 = x1[i]*w4 + x2[i]*w5 + x3[i]*w6 + w11
        y3 = x1[i]*w7 + x2[i]*w8 + x3[i]*w9 + w12
              
        sy1 = (1/(1 + np. exp (-y1)))  #sigmoid
        sy2 = (1/(1 + np. exp (-y2)))
        sy3 = (1/(1 + np. exp (-y3)))
        
        #layer 2
        z1= sy1*w11 + sy2*w21 + sy3*w31 + w01
        z2= sy1*w12 + sy2*w22 + sy3*w32 + w02
        z3= sy1*w13 + sy2*w23 + sy3*w33 + w03
        z4= sy1*w14 + sy2*w24 + sy3*w34 + w04
        
        sz1 = (1/(1 + np. exp (-z1))) #sigmoid
        sz2 = (1/(1 + np. exp (-z2)))
        sz3 = (1/(1 + np. exp (-z3)))
        sz4 = (1/(1 + np. exp (-z4)))
        
        #Last layer
        output = sz1*w111 + sz2*w222 + sz3*w333 + sz4*w444 + w000
        
        sop = (1/(1 + np. exp (-output))) #sigmoid
        
        #error
        err = (sop - op[i])**2 
        
        #back propogation *********************************************
        #layer 1------------------------------------------------------------------
        dw000 = lr*(op[i] - sop) #learning bias
        
        dw111 = err*(sop*(1-sop))*sz1*lr
        w111 = w111 + dw111
        
        dw222 = err*(sop*(1-sop))*sz2*lr
        w222 = w222 + dw222
        
        dw333 = err*(sop*(1-sop))*sz3*lr
        w333 = w333 + dw333
        
        dw444 = err*(sop*(1-sop))*sz4*lr
        w444 = w444 + dw444
        
        #layer 2
        dw01 = lr*(op[i] - sz1) #learning bias (z1) -------------
        
        dw11 = err*(sz1*(1-sz1))*sy1*lr *dw111
        w11 = w11 + dw11
        
        dw21 = err*(sz1*(1-sz1))*sy2*lr *dw111
        w21 = w21 + dw21
        
        dw31 = err*(sz1*(1-sz1))*sy3*lr *dw111
        w31 = w31 + dw31
        
        dw02 = lr*(op[i] - sz2) #learning bias (z2) -----------
        
        dw12 = err*(sz2*(1-sz2))*sy1*lr *dw222
        w12 = w12 + dw12
        
        dw22 = err*(sz2*(1-sz2))*sy2*lr *dw222
        w22 = w22 + dw22
        
        dw32 = err*(sz2*(1-sz2))*sy3*lr *dw222
        w32 = w32 + dw32
        
        dw03 = lr*(op[i] - sz3) #learning bias (z3) -----------
        
        dw13 = err*(sz3*(1-sz3))*sy1*lr *dw333
        w13 = w13 + dw13
        
        dw23 = err*(sz3*(1-sz3))*sy2*lr *dw333
        w23 = w23 + dw23
        
        dw33 = err*(sz3*(1-sz3))*sy3*lr *dw333
        w33 = w33 + dw33
        
        dw04 = lr*(op[i] - sz4) #learning bias (z4) -----------
        
        dw14 = err*(sz4*(1-sz4))*sy1*lr *dw444
        w14 = w14 + dw14 
        
        dw24 = err*(sz4*(1-sz4))*sy2*lr *dw444
        w24 = w24 + dw24
        
        dw34 = err*(sz4*(1-sz4))*sy3*lr *dw444
        w34 = w34 + dw34
        
        dw10 = lr*(op[i] - sy1) #learning bias (y1) -----------
        
        dw1 = err*(sy1*(1-sy1))*x1[i]*lr*(dw11+dw111)*(dw12+dw222)*(dw13+dw333)*(dw14+dw444)
        w1 = w1 + dw1
        
        dw2 = err*(sy1*(1-sy1))*x2[i]*lr*(dw11+dw111)*(dw12+dw222)*(dw13+dw333)*(dw14+dw444)
        w2 = w2 + dw2
        
        dw3 = err*(sy1*(1-sy1))*x3[i]*lr*(dw11+dw111)*(dw12+dw222)*(dw13+dw333)*(dw14+dw444)
        w3 = w3 + dw3
        
        dw11 = lr*(op[i] - sy2) #learning bias (y2) -----------
        
        dw4 = err*(sy2*(1-sy2))*x1[i]*lr*(dw21+dw111)*(dw22+dw222)*(dw23+dw333)*(dw24+dw444)
        w4 = w4 + dw4
        
        dw5 = err*(sy2*(1-sy2))*x2[i]*lr*(dw21+dw111)*(dw22+dw222)*(dw23+dw333)*(dw24+dw444)
        w5 = w5 + dw5
        
        dw6 = err*(sy2*(1-sy2))*x3[i]*lr*(dw21+dw111)*(dw22+dw222)*(dw23+dw333)*(dw24+dw444)
        w6 = w6 + dw6
        
        dw11 = lr*(op[i] - sy3) #learning bias (y3) -----------
        
        dw7 = err*(sy3*(1-sy3))*x1[i]*lr*(dw31+dw111)*(dw32+dw222)*(dw33+dw333)*(dw34+dw444)
        w7 = w7 + dw7
        
        dw8 = err*(sy3*(1-sy3))*x2[i]*lr*(dw31+dw111)*(dw32+dw222)*(dw33+dw333)*(dw34+dw444)
        w8 = w8 + dw8
        
        dw9 = err*(sy3*(1-sy3))*x3[i]*lr*(dw31+dw111)*(dw32+dw222)*(dw33+dw333)*(dw34+dw444)
        w9 = w9 + dw9
        #******************************************************
        count = count + 1
        
        
        
        
        


# In[117]:


print("the final equation is",sop)
print("the final weights are")
print(w111)
print(w222)
print(w333)
print(w444)


# In[ ]:





# In[ ]:




