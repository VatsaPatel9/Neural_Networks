#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
#import tensorflow.keras.Model
# import tensorflow as tf
# tf.keras.Model()
#import tensorflow.compat.v1 as tf

mnist= tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(y_train)

x_train = tf.keras.utils.normalize(x_train, axis = 1) #it will normalize the data to 0 & 1 from numbers 
x_test = tf.keras.utils.normalize(x_test, axis = 1) #easy for nw to learn

model= tf.keras.Sequential() #model will work sequentially #defining the model

model.add(tf.keras.layers.Flatten()) #it flattens the model #input layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #hidden layer
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #hidden layer
#model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) #hidden layer

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) #finaly layer

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',mertics= ['accuracy'])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['mae', 'acc'])
model.fit(x_train, y_train, epochs = 4)


# In[3]:


val_loss,mae,val_acc = model.evaluate(x_test, y_test)
print("Loss ", val_loss)
print("Mean Absoulte Error (mae)")
print("Accuracy ",val_acc)


# In[7]:


# import matplotlib.pyplot as plt

# plt.imshow(x_train[2], cmap = plt.cm.binary) #color map #
# plt.show()
# print(x_train [2])


# In[4]:


model.save('MNIST_abc.model')
new_model = tf.keras.models.load_model('MNIST_abc.model')

predictions = new_model.predict(x_test)
print(predictions)
#print(np.argmax(predicitions[0]))


# In[5]:


import numpy as np

print(np.argmax(predictions[2]))


# In[6]:


plt.imshow(x_test[2])
plt.show()

