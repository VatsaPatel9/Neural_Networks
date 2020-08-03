#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import tensorflow as tf
import tensorflow as tf
import numpy as np 
import datetime
#for tensorflow.examples.tutorials.mnist import input_data
mnist = tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)

#mnist = input_data.read_data_sets("MNIST_data/", one_hot= True)


# In[3]:


import tensorflow.compat.v2 as tf

import tensorflow_datasets as tfds
tfds.disable_progress_bar()
ds_train = tfds.load(name="mnist", split="train")


# In[4]:


tf.compat.v1.disable_eager_execution()

input_size = 784
output_size = 10
hiddent_layer_size = 50

inputs =  tf.compat.v1.placeholder(tf.float32, [None, input_size])
targets =  tf.compat.v1.placeholder(tf.float32, [None, output_size])

weight_1 =  tf.compat.v1.get_variable("weights_1", [input_size,hiddent_layer_size])
biases_1 = tf.compat.v1.get_variable("biases_1", [hiddent_layer_size])
output_1 = tf.nn.relu(tf.matmul(inputs,weight_1)+biases_1)
print(output_1)

weight_2 =  tf.compat.v1.get_variable("weights_2", [hiddent_layer_size,hiddent_layer_size])
biases_2 = tf.compat.v1.get_variable("biases_2", [hiddent_layer_size])
output_2 = tf.nn.relu(tf.matmul(output_1,weight_2)+biases_2)
print(output_2)

weight_3 =  tf.compat.v1.get_variable("weights_3", [hiddent_layer_size,output_size])
biases_3 = tf.compat.v1.get_variable("biases_3", [output_size])
output = tf.matmul(output_2,weight_3)+biases_3
print(output)

loss = tf.nn.softmax_cross_entropy_with_logits (logits = output, labels = targets )
print(loss)
mean_loss= tf.math.reduce_mean(loss)
print(mean_loss)

optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)
print(optimizer)

output_equal_target = tf.math.equal(tf.math.argmax(output,1),tf.math.argmax(targets,1))
print(output_equal_target)

accrucy= tf.math.reduce_mean(tf.cast(output_equal_target, tf.float32))
print(accrucy)

initializer = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.InteractiveSession()
sess.run(initializer)

#batch_size=100

#batches_number= mnist.train._num_examples //batch_size

max_epochs = 15

prev_validation_loss = 999999.


# In[5]:


for = epoch_counter in range (max_epochs):
    curr_epoch_loss = 0.
    
    for batch_counter in range (batches_number):
        input_batch, target_batch = mnist.train.next_batch(batch_size)
        _,batch_loss = sess.run([optimizer,mane_loss],feed_dict={inputs: input_batch, target: target_batch})
        curr_epoch_loss += batch_loss
        
    curr_epoch_loss /= batches_number
    
    input_batch, target_batch = mnist.validation.next_batch(mnist.validation._num_examples)
    
    validation_loss, validation_accuracy = sess.run([mean_loss, accuracy], feed_dict={inputs: input_batch, targets: target_batch})
    
    print('Epoch'+str(epoch_counter+1)+ '. Training loss:'+'{0:.3f}'.format(curr_epoch_loss))

