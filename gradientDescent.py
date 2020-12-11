#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# %%
#variable
x1=tf.Variable(tf.constant([-4,4],tf.float32),tf.float32)
#function
y1=tf.reduce_sum(tf.square(x1))

#%%
#create a session
session = tf.Session()
session.run(tf.global_variables_initializer())


# %%
#gradient descent
opt1=tf.train.GradientDescentOptimizer(0.25).minimize(y1)


# %%
for i in range(100):
    session.run(opt1)
    print('x',session.run(x1))
    print('y',session.run(y1))
session.close()
# %%
# Adam algorithm
# variable
x2=tf.Variable(tf.constant([[4],[3]],tf.float32),tf.float32)
w=tf.constant([[1,2]],tf.float32)
# function
y2=tf.matmul(w,tf.square(x2))


# %%
session=tf.Session()
session.run(tf.global_variables_initializer())


# %%
opt2=tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(y2)

# %%
for i in range(100):
    session.run(opt2)
    print("x",session.run(x2))
    print("y",session.run(y2))

# %%
