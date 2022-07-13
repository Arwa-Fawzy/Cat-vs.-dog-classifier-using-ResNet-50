
# In[1]:


# required libraries 
import os
from os import listdir
import PIL
from PIL import Image
import pandas as pd
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.initializers import glorot_uniform
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add, AveragePooling2D, GlobalMaxPooling2D,MaxPool2D
from keras.callbacks import EarlyStopping,ModelCheckpoint
import h5py
import jupyterlab_hdf




#training the model
images=os.listdir("E:/CS/Robotics club/New folder/cat vs dog/train")
#it depends on your path's file
categories=[]
for image in images:
    category=image.split('.')[0]
    if category=='dog':
        categories.append(1)
    else:
        categories.append(0)
df=pd.DataFrame({
    'imagefilename':images,
    'category':categories
})
# as mentioned, the classfier will have 1 for dogness and 0 for catness 

#some augmentations
#augmentation means increasing the data or becoming larger rather than collecting more images manually 

train_datagen = ImageDataGenerator(zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15)
test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory("E:/CS/Robotics club/New folder/cat vs dog/train",target_size=(224, 224),batch_size=32,shuffle=True,class_mode='binary')
test_generator = test_datagen.flow_from_directory("E:/CS/Robotics club/New folder/cat vs dog/test1",target_size=(224,224),batch_size=32,shuffle=False,class_mode='binary')

#reading the image into an array & as a gray scale
#resize the images into (224,224) as ResNet50 requires this size 
train = 'E:/CS/Robotics club/New folder/cat vs dog/train'
path = os.path.join('E:/CS/Robotics club/New folder/cat vs dog','E:/CS/Robotics club/New folder/cat vs dog/train')
for p in os.listdir(path):
    category = p.split(".")[0]
    img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
    new_img_array = cv2.resize(img_array, dsize=(224, 224))
    plt.imshow(new_img_array,cmap="gray")
    break
    
a = [] #training array
b = [] #target array
convert = lambda category : int(category == 'dog')
def create_test_data(path):
    for p in os.listdir(path):
        category = p.split(".")[0]
        category = convert(category)
        img_array = cv2.imread(os.path.join(path,p),cv2.IMREAD_GRAYSCALE)
        new_img_array = cv2.resize(img_array, dsize=(224, 224))
        a.append(new_img_array)
        b.append(category)



# In[2]:


#Before implementing ResNet50, it is indispensable to consider two parts: identity block and convolution block 

def identity_block(X, f, filters, stage, block):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters

    X_shortcut = X
   
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])# SKIP Connection
    X = Activation('relu')(X)

    return X


# In[3]:


def convolutional_block(X, f, filters, stage, block, s=2):
   
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# In[4]:


def ResNet50(input_shape=(224, 224, 3)):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)
    
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


# In[5]:


base_model = ResNet50(input_shape=(224, 224, 3))
# without fully connected layer (the former ones were pre-trained data) - Transfer learning criteria


# In[6]:


headModel = base_model.output
headModel = Flatten()(headModel)
headModel=Dense(256, activation='relu', name='fc1',kernel_initializer=glorot_uniform(seed=0))(headModel) 
headModel=Dense(128, activation='relu', name='fc2',kernel_initializer=glorot_uniform(seed=0))(headModel)
headModel = Dense( 1,activation='sigmoid', name='fc3',kernel_initializer=glorot_uniform(seed=0))(headModel)


# In[7]:


model = Model(inputs=base_model.input, outputs=headModel) #input of this model is placed at the last layer


# In[8]:


model.summary()


# In[9]:


#finetuning: directing the model specifically on the last layers and freezing the pre-trained data 
h5 = pd.HDFStore('E:\CS\Robotics club\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 (1)', 'r')
h5.open()
base_model.load_weights("E:\CS\Robotics club\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 (1)") 


# In[ ]:


for layer in base_model.layers:
    layer.trainable = False #freezing the trainable data via setting false value for its layers (up to last maxpooling layer)


# In[ ]:


for layer in model.layers:
    print(layer, layer.trainable)
    


# In[ ]:


#Earlystopping can avoid underfitting and overfitting through determining the approciate epochs number 
es=EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
#Earlystopping ends the epochs once the model stoped improving its validation data
#Earlystopping has four arguments: mode, monitor, verbose, patience 


# In[ ]:


mc = ModelCheckpoint('E:\CS\Robotics club\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 (1)', monitor='val_accuracy', mode='
# ModelCheckpoint will save the best validation set result for subsequent training or usage, preventing from approaching worse accuracy
H = model.fit_generator(train_generator,validation_data=test_generator,epochs=100,verbose=1,callbacks=[mc,es])
model.load_weights("E:\CS\Robotics club\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 (1)")
model.evaluate_generator(test_generator) #printing the final percentage of accuracy


# In[ ]:





# In[ ]:




