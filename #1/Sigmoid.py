		#1

This is the first step of the Neural Network, where a single neuron manually and also used sigmoid function in it. The inputs are (X1,X2,X3) and the outputs is Y

X1	X2	X3	Y (output)
0	0	0	1
0	1	1	1
1	0	1	1
1	1	1	1
1	1	0	1
4	0	4	0
0	4	4	0
4	4	4	0






#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from matplotlib import pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(y):
    return(y*(1-y))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
k = np.linspace(-6,6,5)
m= np.linspace(-6,6,num=5)

# learning rate
l=0.01

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
w = random.uniform(-2.0,2)
print("w1",w1)
print("w2",w2)
print("w3",w3)
print("w4",w)
ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

for i in range(0,8):
    if y[i]>0:
        # mark positive output using a blue circle
        ax.scatter(x1[i],x2[i],x3[i],marker="s",c="b")
    else:
        # mark negative output using a red square
        ax.scatter(x1[i],x2[i],x3[i],marker="o",c="r")
        

count=1
while(count < 1000):
    print ("Training interation ", count)

    a0=np.zeros(8)

    for i in range(0,8):
        op = (x1[i] * w1 )+ (x2[i] * w2 ) + (x3[i] * w3 ) + w
        ao[i] = (1/(1+np.exp(op)))
        if (ao[i]==y[i]):
            continue
        err = (y[i]-ao[i])**2
        der = (y[i]-a0[i])
        dw1 = float(-2 * l * x1[i]*(sigmoid(a0[i]))*der)
        dw2 = float(-2 * l * x2[i]*(sigmoid(a0[i]))*der)
        dw3 = float(-2 * l * x3[i]*(sigmoid(a0[i]))*der)
        dw = float(-2 * l * (sigmoid(a0[i]))*der)
       
        w1+=dw1
        w2+=dw2
        w3+=dw3
        w+=dw
        
        
    
    count = count + 1
X, Y = np.meshgrid(k, m)
Z = -(X*w1+Y*w2+w)/w3
print('The equation is {0}x + {1}y + {2}z = {3}'.format(w1, w2, w3, w))#Equation of plane 
ax.scatter3D(x1,x2,x3,c = y)
ax.plot_surface(X,Y,Z)        
    
plt.show()

        
        

