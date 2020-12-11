#%%
import numpy as np
import os, shutil
# %%
# prepare dataset
original_dataset_dir = r'G:\Work\file\archive\training_set\training_set'

base_dir = r'G:\Work\file\archive\cat_dog_small'
os.mkdir(base_dir)
# %%
train_dir = os.path.join(base_dir,'train')
os.mkdir(train_dir)
valid_dir = os.path.join(base_dir,'valid')
os.mkdir(valid_dir)
test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)
# %%
for directory in ['train', 'valid', 'test']:
    for category in ['cats', 'dogs']:
        os.mkdir(base_dir+"\{}".format(directory)+"\{}".format(category))
# %%
sizes = {'train':[0,1000],'valid':[1000,1500],'test':[1500,2000]}
for directory in ['train', 'valid', 'test']:
    for category in ['cat', 'dog']:
        fnames = ['{}.{}.jpg'.format(category, i+1) for i in range(sizes[directory][0],sizes[directory][1])]
        for fname in fnames:
            src = os.path.join(original_dataset_dir+"\{}s".format(category),fname)
            dst = os.path.join(base_dir+"\{}".format(directory)+"\{}s".format(category),fname)
            shutil.copyfile(src,dst)
# %%
