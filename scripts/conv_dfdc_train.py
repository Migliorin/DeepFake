FRAMES = 290
PATH = "../datasets/dfdc_train_part_1"
IMG_SIZE = 224
BATCH_SIZE = 18
EPOCHS = 10

import os


os.environ["CUDA_VISIBLE_DEVICES"]=""


import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Flatten, BatchNormalization, Dropout,Conv3D, MaxPooling3D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import optimizers

import cv2


from sklearn.model_selection import train_test_split


np.random.seed(30)
tf.random.set_seed(30)



def pre_process_video(path_video:str,img_index:list,resize:tuple)->list:
    frames = []
    vidcapture = cv2.VideoCapture(path_video)
    index = 0
    j = 0
    while(vidcapture.isOpened()):
        rent, frame = vidcapture.read()
        if(not rent):
            break
        else:
            if(len(img_index) - 1 < j):
                break
            else:
                if(index == img_index[j]):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame,resize)
                    frames.append(frame)
                    j += 1

                index += 1

    frames = np.array(frames)
    frames = (frames / 255)
    frames = frames.reshape(frames.shape[0],frames.shape[1],frames.shape[2],1)
    return frames


def getImgTensor(n_frames:int)->list:
    img_idx = np.round(np.linspace(0, FRAMES, n_frames)).astype(int)
    return [img_idx, IMG_SIZE, IMG_SIZE, 1]


def getBatchData(train_metadata,batch,batch_size,img_tensor)->tuple:
    [len_frames,width,length] = [len(img_tensor[0]),img_tensor[1], img_tensor[2]] # dimensions
    img_idx = img_tensor[0] # array index of frames
    
    batch_data = np.zeros((batch_size,len_frames,width,length,1)) # batch data that will pass forward
    batch_labels = np.zeros((batch_size,2)) # batch labels that will pass forward
    
    #############################################################
    # Here is how the batch data is split by callback
    if(((batch+1)*batch_size) <= train_metadata.shape[0]):
        train_metadata_ = train_metadata.iloc[
            batch*batch_size:(batch+1)*batch_size,
            :
        ]
    else:
        train_metadata_ = train_metadata.iloc[
            batch*batch_size:,
            :
        ]
    
    #############################################################
    video_posi = 0
    name_list = train_metadata_['name'].to_list()
    label_list = train_metadata_["label"].to_list()
    
    for name,label in zip(name_list,label_list):
        path_ = f"{PATH}/{name}"
        batch_data[video_posi] = pre_process_video(path_,
                                          img_idx,
                                          (width,length))
        
        if(label == "FAKE"):
            batch_labels[video_posi][0] = 1
        else:
            batch_labels[video_posi][1] = 1
            
        video_posi += 1
            
    return batch_data, batch_labels

def generator(train_metadata, batch_size, img_tensor):
    while True:
        if(len(train_metadata["name"])%batch_size == 0):
            num_batches = int(len(train_metadata["name"])/batch_size)
        else:
            num_batches = int(len(train_metadata["name"])/batch_size) + 1
        
        for batch in range(num_batches): # we iterate over the number of batches
            yield getBatchData(train_metadata,batch,batch_size,img_tensor)

def make3dFilter(x):
    return tuple([x]*3)

def make2dFilter(x):
    return tuple([x]*2)

#write your model here
def defineModel(img_tensor):
    inputShape = (len(img_tensor[0]), img_tensor[1], img_tensor[2], img_tensor[3])
    print(inputShape)
    model = Sequential([
        Conv3D(16, make3dFilter(5), activation='relu', input_shape=inputShape),
        MaxPooling3D(make3dFilter(2), padding='same'),
        BatchNormalization(),

        Conv3D(32, make3dFilter(3), activation='relu'),
        MaxPooling3D(pool_size=(1,2,2), padding='same'),
        BatchNormalization(),

        Conv3D(64, make3dFilter(3), activation='relu'),
        MaxPooling3D(pool_size=(1,2,2), padding='same'),
        BatchNormalization(),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.25),

        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model





train_metadata = pd.read_json(f"{PATH}/metadata.json")
train_metadata = train_metadata.T
train_metadata.reset_index(inplace=True)
train_metadata.rename({"index":"name"},axis=1,inplace=True)

img_tensor = getImgTensor(25)
model = defineModel(img_tensor)

train, test = train_test_split(train_metadata,test_size=0.33,random_state=42,stratify=train_metadata["label"])

train_generator = generator(train, BATCH_SIZE, img_tensor)
val_generator = generator(test, BATCH_SIZE, img_tensor)

if (train.shape[0]%BATCH_SIZE) == 0:
    steps_per_epoch = int(train.shape[0]/BATCH_SIZE)
else:
    steps_per_epoch = (train.shape[0]//BATCH_SIZE) + 1

if (test.shape[0]%BATCH_SIZE) == 0:
    validation_steps = int(test.shape[0]/BATCH_SIZE)
else:
    validation_steps = (test.shape[0]//BATCH_SIZE) + 1


import datetime

curr_dt_time = datetime.datetime.now()

model_name = 'model_init' + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
    
if not os.path.exists(model_name):
    os.mkdir(model_name)

filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

LR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=4)

# callbacks_list = [checkpoint, LR]
callbacks_list = [LR]


model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1, 
            callbacks=callbacks_list, validation_data=val_generator, 
            validation_steps=validation_steps, class_weight=None, workers=1, initial_epoch=0)

