# This is an introduction to the TensorFlow where the efforts of creating the neural network is being reduced and everything is done by TensorFlow itself  
# automatically in just few lines of code. Here different types of liberaries are being used like matplotlib, os, keras and tensorflow

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras,os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#import numpy as np






# # Load the data

# In[2]:


os.chdir(r"path")
retval = os.getcwd() #returns current working directory 
print(retval)
train_dir = os.path.join(retval, 'train')
print(train_dir)
base_dir = os.getcwd()
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
print(train_cats_dir)

# Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cats_fnames = os.listdir(train_cats_dir)
train_dogs_fnames=  os.listdir(train_dogs_dir)
print(train_cats_fnames[:10])
print(train_dogs_fnames[:10])


# # Understanding the data

# In[3]:


#print the number of cats and dogs in the directory
print('total training cats images :', len(os.listdir(train_cats_dir))) 
print('total training dogs images :', len(os.listdir(train_dogs_dir)))
print('total validation cats images :', len(os.listdir(validation_cats_dir)))
print('total validation dogs images :', len(os.listdir(validation_dogs_dir)))


# # Visualize training images

# In[4]:


nrows = 4 
ncols = 4
pic_index = 0 # Index for iterating over images

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)
pic_index+=8
next_cat_pix = [os.path.join(train_cats_dir, fname)
for fname in train_cats_fnames[ pic_index-8:pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, fname)
for fname in train_dogs_fnames[ pic_index-8:pic_index]]
for i, img_path in enumerate(next_cat_pix+next_dog_pix):# Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)
    img = mpimg.imread(img_path) 
    plt.imshow(img)
#plt.show()


# # Data Augmentation 

# In[5]:


train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

train_data_gen = train_datagen.flow_from_directory(batch_size=160,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(150, 150),
                                                     class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]


test_datagen = ImageDataGenerator(rescale=1./255)
test_data_gen = test_datagen.flow_from_directory(batch_size=100,
                                                 directory=validation_dir,
                                                 target_size=(150, 150),
                                                 class_mode='binary')



# # Creating a model

# In[6]:


model = Sequential()
model.add(Conv2D(input_shape=(150,150,3),filters=16,kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(input_shape=(150,150,3),filters=32,kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(input_shape=(150,150,3),filters=64,kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(input_shape=(150,150,3),filters=128,kernel_size=(3,3), activation="relu"))
model.add(MaxPool2D(pool_size=(2,2)))

tf.keras.layers.Dropout(0.20)
model.add(Flatten())
model.add(Dense(units=300,activation="relu"))
model.add(Dense(units=300,activation="relu"))
model.add(Dense(units=1,activation="sigmoid"))

#model.compile(loss='binary_crossentropy',optimizer='adam',learning_rate=0.001,metrics=['mae', 'acc'])



model.summary()


# # Compile the model

# In[7]:


from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(lr=0.001),
 loss='binary_crossentropy',
 metrics = ['acc'])


# # Train the model

# In[8]:


stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',min_delta=0.2, verbose=1, patience=3)

history = model.fit(train_data_gen,validation_data=test_data_gen,steps_per_epoch=100,callbacks=[stop],
                         epochs=13,validation_steps=50,verbose=1)


# # Visualize the training

# In[10]:


acc = history.history['acc']
print(acc)
val_acc = history.history['val_acc']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(13) #change it ccording the early stopping 

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# # Save the model

# In[1]:


os.chdir(r"C:\Users\19378\Desktop\testing_C&D")
testing = os.getcwd() #returns current working directory 
print(testing)

test_dir = os.path.join(testing, 'c&d')

print('total test cats&dogs images :', len(os.listdir(test_dir))) 


test_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_generator = test_datagen.flow_from_directory(testing,class_mode='binary',target_size=(150, 150))


print(test_generator)



# In[21]:


model.save('C&D.model')
new_model = tf.keras.models.load_model('C&D.model')

predictions = new_model.predict(test_generator)
print(predictions)
if predictions < 0.5:
    print(predictions)
    print('cat')
    
elif predictions >=0.5:
    predictions = 1
    print(predictions)
    print('dog')


# In[ ]:




