 #2

# Used a fully connected two layer neural network to classify the same data set of single neuron. 
# The first layer (input layer) has three neurons and the second layer (output layer) has only one neuron.
# Using the sigmoid activation function and the same learning rules that was discussed in the class. Printed the input, Ws,  and output values after training. 




#!/usr/bin/env python
# coding: utf-8

# In[29]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D


    
# learning rate

lr=0.0001

#  input x1
x1 = [0,0,1,1,1,4,0,4]

#  input x2
x2 = [0,1,0,1,1,0,4,4]

# input x3
x3 = [0,1,1,1,0,4,4,4]

#  expected output
y = [1,1,1,1,1,0,0,0]



# Generating random weights 
w1 = random.uniform(-2.0,2)
w2 = random.uniform(-2.0,2)
w3 = random.uniform(-2.0,2)
w4 = random.uniform(-2.0,2) #bias
w5 = random.uniform(-2.0,2)
w6 = random.uniform(-2.0,2)
w7 = random.uniform(-2.0,2)
w8 = random.uniform(-2.0,2) #bias
w9 = random.uniform(-2.0,2)
w10 = random.uniform(-2.0,2)
w11 = random.uniform(-2.0,2)
w12 = random.uniform(-2.0,2) #bais
w13 = random.uniform(-2.0,2)
w14 = random.uniform(-2.0,2)
w15 = random.uniform(-2.0,2)
w16 = random.uniform(-2.0,2) #final w


print("w1",w1)
print("w2",w2)
print("w3",w3)
print("w4",w4)


# Different neurons 

#y4 = y1*w13 + y2*w14 + y3*w15 + w16
count=0

while(count < 1000):
    
    #ao=np.zeros(8)
    ao=(-2*(y[i]-(float(1/(1+np.exp(y4))))))
    for i in range(0,8):
        y1 = x1[i]*w1+ x2[i]*w2 + x3[i]*w3 + w4
        y2 = x1[i]*w5 + x2[i]*w6 + x3[i]*w7 + w8
        y3 = x1[i]*w9 + x2[i]*w10 + x3[i]*w11 + w12
        z1 = (1/(1 + np . exp (-y1)))
        z2 = (1/(1 + np . exp (-y2)))
        z3 = (1/(1 + np . exp (-y3)))
        
        if(ao == y1 and ao == y2 and ao == y3 ):
            continue
            
        err = 2*(y[i]-ao)
        #err_2 = 2*(y2[i]-ao[i])
        #err_3 = 2*(y3[i]-ao[i])
            
#             dw1 = float(-2 * l * x1[i]*(sigmoid(a0[i]))*der)
#             dw2 = float(-2 * l * x2[i]*(sigmoid(a0[i]))*der)
#             dw3 = float(-2 * l * x3[i]*(sigmoid(a0[i]))*der)
#             dw4 = float(-2 * l * (sigmoid(a0[i]))*der)
#             dw5 = float(-2 * l * x1[i]*(sigmoid(a0[i]))*der)
#             dw6 = float(-2 * l * x2[i]*(sigmoid(a0[i]))*der)
#             dw7 = float(-2 * l * x3[i]*(sigmoid(a0[i]))*der)
#             dw8 = float(-2 * l * (sigmoid(a0[i]))*der)
#             dw9 = float(-2 * l * x1[i]*(sigmoid(a0[i]))*der)
#             dw10 = float(-2 * l * x2[i]*(sigmoid(a0[i]))*der)
#             dw11 = float(-2 * l * x3[i]*(sigmoid(a0[i]))*der)
#             dw12 = float(-2 * l * (sigmoid(a0[i]))*der)
#             dw13 = float(-2 * l * x1[i]*(sigmoid(a0[i]))*der)
#             dw14 = float(-2 * l * x2[i]*(sigmoid(a0[i]))*der)
#             dw15 = float(-2 * l * x3[i]*(sigmoid(a0[i]))*der)
#             dw16 = float(-2 * l * (sigmoid(a0[i]))*der)
###
        dw13 = err*(y1*(1-y1))*z1*lr
        w13 = w13 + dw13
        dw14 = err*(y2*(1-y2))*z2*lr
        w14 = w14 + dw14
        dw15 = err*(y3*(1-y3))*z3*lr
        w15 = w15 + dw15
        dw16 = lr*(y[i]-ao)
        w16 = w16 + dw16
            
        dw1 = w13*(1-z1)*x1[i]
        w1 = w1 + dw1
        dw2 = w13*(1-z1)*x2[i]
        w2 = w2 + dw2
        dw3 = w13*(1-z1)*x3[i]
        w3 = w3 + dw3
        dw4 = w13*(1-z1)
        w4 = w4 + dw4
        dw5 = w14*(1-z2)*x1[i]
        w5 = w5 + dw5
        dw6 = w14*(1-z2)*x2[i]
        w6 = w6 + dw6
        dw7 = w14*(1-z2)*x3[i]
        w7 = w7 + dw7
        dw8 = w14*(1-z2)
        w8 = w8 + dw8
        dw9 = w15*(1-z3)*x1[i]
        w9 = w9 + dw9
        dw10 = w15*(1-z3)*x2[i]
        w10 = w10 + dw10
        dw11 = w15*(1-z3)*x3[i]
        w11 = w11 + dw11
        dw12 = w15*(1-z3)
        w12 = w12 + dw12
            
            
#         w1+=dw1
#         w2+=dw2
#         w3+=dw3
#         w4+=dw4
#         w5+=dw5
#         w6+=dw6
#         w7+=dw7
#         w8+=dw8
#         w9+=dw9
#         w10+=dw10
#         w11+=dw11
#         w12+=dw12
#         w13+=dw13
#         w14+=dw14
#         w15+=dw15
#         w16+=dw16
            
    count = count + 1
         
        
    y4 = (x1[i]* w13)+ (x2[i]* w14) + (x3[i] * w15) + w16
        
    if count==1000:
        print("the final equation is",y4)
        break
    else:
        print("the current equation is",y4)
        
        


# In[ ]:




