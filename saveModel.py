#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(r"G:\\workspace\pythonWanshenghua\TF")

# %%
v1=tf.Variable(tf.constant([1,2,3],tf.float32),dtype=tf.float32,name="v1")
v2=tf.Variable(tf.constant([4,5],tf.float32),dtype=tf.float32,name="v2")


# %%
saver=tf.train.Saver()

# %%
session=tf.Session()
session.run(tf.global_variables_initializer())


# %%
# save the variables
save_path=saver.save(session,"./model/model.ckpt")
session.close()

# %%
v1=tf.Variable([11,12,13],dtype=tf.float32,name="v1")
v2=tf.Variable([15,16],dtype=tf.float32,name="v2")

# %%
saver=tf.train.Saver()
with tf.Session() as session:
    saver.restore(session,"./model/model.ckpt")
    print(session.run(v1))
    print(session.run(v2))
session.close()

# %%
