#%%
import os
import numpy as np
from keras import layers
from keras import models
import cv2
# %%
# 准备数据
import os,shutil
base_dir = r'G:\Work\file\archive\rotate'
ori_dir = r'G:\Work\file\archive\training_set\training_set'
#%%
train_dir = os.path.join(base_dir,'train')
#os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation')
#os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test')
#os.mkdir(test_dir)
# %%
for rot in [0,90,180,270]:
    for name in ['train','validation','test']:
        os.mkdir(base_dir + '\{}\{}'.format(name,rot))
# %%
nums = {'train':[0,1000],'validation':[1000,1500],'test':[1500,2000]}
for rot in [90,180,270]:
    for name in ['train','validation','test']:
        for cat in ['cat','dog']:
            for i in range(nums[name][0],nums[name][1]):
                img = cv2.imread(os.path.join(ori_dir,'{}.{}.jpg'.format(cat,i+1)),cv2.IMREAD_COLOR)
                img = cv2.resize(img, (150,150), interpolation=cv2.INTER_CUBIC)
                h,w = img.shape[:2]
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, rot, 1.0)
                rotatedimg = cv2.warpAffine(img, M, (w,h))
                cv2.imwrite(os.path.join(base_dir,'{}\{}\{}_{}.jpg'.format(name,rot,cat,i+1)),rotatedimg)
# %%
#将猫狗分类的小型卷积神经网络实例化
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(256,(3,3),activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.summary()
# %%
from keras import optimizers

model.compile(loss='categorical_crossentropy',
             optimizer = optimizers.RMSprop(lr=1e-4),
             metrics = ['acc'])
# %%
#使用ImageDataGenerator从目录中读取图像
#ImageDataGenerator可以快速创建Python生成器，能够将硬盘上的图像文件自动转换为预处理好的张量批量
from keras.preprocessing.image import ImageDataGenerator

#将所有图像乘以1/255缩放
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (150,150),
    batch_size = 40,
    class_mode = 'categorical'  #因为使用了binary_crossentropy损失，所以需要用二进制标签
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'categorical'
)
# %%
for data_batch,labels_batch in train_generator:
    print('data batch shape:',data_batch.shape)
    print('labels batch shape:',labels_batch.shape)
    break
# %%
#利用批量生成器拟合模型
history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = 100,
    validation_data = validation_generator,
    validation_steps = 50#需要从验证生成器中抽取50个批次用于评估
)

#保存模型
model.save('model\cats_and_dogs_rotate_1.h5')
# %%
#绘制损失曲线和精度曲线
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training_acc')
plt.plot(epochs,val_acc,'b',label='Validation_acc')
plt.title('Traing and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation_loss')
plt.title('Traing and validation loss')
plt.legend()

plt.show()
# %%
