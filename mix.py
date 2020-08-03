#!/usr/bin/env python
# coding: utf-8

# In[58]:


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
l=0.1

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
        
close_flag=True
count=1

while(count < 100):
    print ("Training interation ", count)

    ao=np.zeros(8)

    for i in range(0,8):
        op = (x1[i] * w1 )+ (x2[i] * w2 ) + (x3[i] * w3 ) + w4
        ao[i] = (1/(1+np.exp(op)))
        if (ao[i]==eo[i]):
            continue
        closeflag=False
        err = (eo[i]-ao[i])**2
        der = (ao[i]*(1-ao[i]))
        nw1 = float(-2 * lr * x1[1]*(ao[i]*(1-ao[i]))*err)
        nw2 = float(-2 * lr * x2[1]*(ao[i]*(1-ao[i]))*err)
        nw3 = float(-2 * lr * x3[1]*(ao[i]*(1-ao[i]))*err)
        nw4 = float(-2 * lr * (ao[i]*(1-ao[i]))*err)
        w1+=nw1
        w2+=nw2
        w3+=nw3
        w4+=nw4
        print (w1)
        print (w2)
        print (w3)
        print (w4)
        ax.scatter3D(x1,x2,x3)

        X, Y = np.meshgrid(k, m)
        Z = -(X*w1+Y*w2+w4)/w3
        ax.scatter3D(x1,x2,x3,c = eo)
        ax.plot_surface(X,Y,Z)


    count = count + 1


    if (closeflag):
        break;

plt.show()



        
        


# In[57]:


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import uniform as uni



   
   
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#  input x1
x1 = [0,0,1,1,1,4,0,4]

#  input x2
x2 = [0,1,0,1,1,0,4,4]

#  input x3
x3 = [0,1,1,1,0,4,4,4]

#  expected output
y = [1,1,1,1,1,0,0,0]

# Generating random weights
w1 = uni(-2.0,3)
w2 = uni(-2.0,3)
w3 = uni(-2.0,3)

# Fixing bias to 1
w = 1

#Learning rate
lr = 0.01

output = []

flag = False
count = 1
k = np.linspace(-6,6,5)
m= np.linspace(-6,6,num=5)

ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("z axis")

for i in range(0,8):
    if y[i]>0:
        # mark positive output using a blue circle
        ax.scatter(x1[i],x2[i],x3[i],marker="o",c="b")
    else:
        # mark negative output using a red square
        ax.scatter(x1[i],x2[i],x3[i],marker="s",c="r")


while count < 2000:
    print("Training interation ", count)

    for i in range(0,8):
        flag = True

        #  weighted sum
        d = (x1[i]*w1) + (x2[i]*w2) + (x3[i]*w3) + w
        # print("weighted sum: ", d)
       
        if(d >=0 ):
            o = 1
        else:
            o = 0
         

        #  change in w1
        dw1 = lr*(x1[i]*(y[i]-o))
        # print("dw1: ", dw1)

        #  change in w2
        dw2 = lr*(x2[i]*(y[i]-o))
       
        #  change in w3
        dw3 = lr*(x3[i]*(y[i]-o))
       
        dw = lr*(y[i]-o)
        w1 = w1 + dw1
        w2 = w2 + dw2
        w3 = w3 + dw3

        # Uncomment this if we want to change bias along with weights
        w = w + dw

        output.insert(i,o)

        # Check to see if change in weights are all zero
        if (dw1 > 0 or dw2 > 0 ):
            flag = False

    # print("Output : ", output)
   
    # Line equation
   
    X, Y = np.meshgrid(k, m)
    Z = -(X*w1+Y*w2+w)/w3
   
    print('The equation is {0}x + {1}y + {2}z = {3}'.format(w1, w2, w3, w))
   
    if ( flag==True and y==output):
        print("Training is complete")
        # plotting a straight line for final boundry
        #ax.plot3d(k,l,d,linestyle='-', c='b')
        break
    else:
        # Plotting a dotted line for each boundry
        ax.plot(X.flatten(),
        Y.flatten(),
        Z.flatten(),  )


    #    Ensure that the next plot doesn't overwrite the first plot
       
       
        output = []
    count = count + 1
   
# print("Final weights: w1=%f , w2=%f, bias=%f" % (w1,w2,w))
print("Total training steps: ", count-1)

ax.view_init(5)

plt.show()


# In[36]:


import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import uniform as uni

closeflag=True

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1=[0,0,1,1,1,4,0,4]
x2=[0,1,0,1,1,0,4,4]
x3=[0,1,1,1,0,4,4,4]
eo=[1,1,1,1,1,0,0,0]
count = 1
k = np.arange(-6,6,1)
m = np.arange(-6,6,1)
ax.scatter3D(x1,x2,x3)

for i in range(0,8):
    if eo[i]>0:
        # mark positive output using a blue circle
        ax.scatter(x1[i],x2[i],x3[i],marker="o",c="b")
    else:
        # mark negative output using a red square
        ax.scatter(x1[i],x2[i],x3[i],marker="s",c="r")


lr=0.1

w1=uni(-2.0,2)
w2=uni(-2.0,2)
w3=uni(-2.0,2)
w4=uni(-2.0,2)

while(count < 100):
    print ("Training interation ", count)

    ao=np.zeros(8)

    for i in range(0,8):
        op = (x1[i] * w1 )+ (x2[i] * w2 ) + (x3[i] * w3 ) + w4
        ao[i] = (1/(1+np.exp(op)))
        if (ao[i]==eo[i]):
            continue
        closeflag=False
        err = (eo[i]-ao[i])**2
        der = (ao[i]*(1-ao[i]))
        nw1 = float(-2 * lr * x1[1]*(ao[i]*(1-ao[i]))*err)
        nw2 = float(-2 * lr * x2[1]*(ao[i]*(1-ao[i]))*err)
        nw3 = float(-2 * lr * x3[1]*(ao[i]*(1-ao[i]))*err)
        nw4 = float(-2 * lr * (ao[i]*(1-ao[i]))*err)
        w1+=nw1
        w2+=nw2
        w3+=nw3
        w4+=nw4
        print (w1)
        print (w2)
        print (w3)
        print (w4)
        ax.scatter3D(x1,x2,x3)

        X, Y = np.meshgrid(k, m)
        Z = -(X*w1+Y*w2+w4)/w3
        ax.scatter3D(x1,x2,x3,c = eo)
        ax.plot_surface(X,Y,Z)


    count = count + 1


    if (closeflag):
        break;

plt.show()


# In[ ]:





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

        
        

