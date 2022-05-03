import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

from tensorflow.keras.layers import Dense, Flatten, Flatten, BatchNormalization, Dropout,Conv3D, MaxPooling3D
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import datetime
import json
import cv2
import re


np.random.seed(30)
tf.random.set_seed(30)


class ImageTensor():
    def __init__(self) -> None:
        ## Todos os path nao podem tem '/' no final
        self.frames = None
        self.n_frames = None
        self.path = None
        self.path_plots = None
        self.img_size = None
        self.batch_size = None
        self.epochs = None
        self.dim_color = None
        self.img_idx = None
        self.switch_image_tensor_default()

    def switch_image_tensor_default(self)->None:
        self.frames = 290
        self.n_frames = 25
        self.path = "../datasets/dfdc_train_part_1"
        self.path_plots = "../plots"
        self.img_size = 224
        self.batch_size = 18
        self.epochs = 10
        self.dim_color = 1
        self.img_idx = np.round(np.linspace(0, self.frames, self.n_frames)).astype(int)

    def switch_image_tensor_49(self)->None:
        self.path = "../datasets/dfdc_train_part_49"

    

    
    def get_atributos(self)->dict:
        return {
            "frames"        : self.frames,
            "n_frames"      : self.n_frames,
            "path"          : self.path,
            "path_plots"    : self.path_plots,
            "img_size"      : self.img_size,
            "batch_size"    : self.batch_size,
            "epochs"        : self.epochs,
            "dim_color"     : self.dim_color,
            "img_idx"       : [int(x) for x in self.img_idx]
        }


class PreProcessVideo():
    def __init__(self,image_tensor:ImageTensor) -> None:
        self.image_tensor = image_tensor

    def pre_process_video(self,path_video)->list:
        frames = []
        vidcapture = cv2.VideoCapture(path_video)
        index = 0
        j = 0
        while(vidcapture.isOpened()):
            rent, frame = vidcapture.read()
            if(not rent):
                break
            else:
                if(len(self.image_tensor.img_idx) - 1 < j):
                    break
                else:
                    if(index == self.image_tensor.img_idx[j]):
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        resize = (self.image_tensor.img_size,self.image_tensor.img_size)
                        frame = cv2.resize(frame,resize)
                        frames.append(frame)
                        j += 1

                    index += 1

        frames = np.array(frames)
        frames = (frames / 255)
        frames = frames.reshape(frames.shape[0],frames.shape[1],frames.shape[2],self.image_tensor.dim_color)
        return frames




class LayersAux(): 
    def make3dFilter(self,x)->tuple:
        return tuple([x]*3)

    def make2dFilter(self,x)->tuple:
        return tuple([x]*2)
    

class BatchData():
    def __init__(self,train_metadata:pd.DataFrame,preProcessVideo:PreProcessVideo) -> None:
        self.train_metadata = train_metadata
        self.preProcessVideo = preProcessVideo

    def getBatchData(self,batch:int)->tuple:
        
        batch_size = self.preProcessVideo.image_tensor.batch_size
        # dimensions
        batch_data = np.zeros((
            batch_size,
            len(self.preProcessVideo.image_tensor.img_idx),
            self.preProcessVideo.image_tensor.img_size,
            self.preProcessVideo.image_tensor.img_size,
            self.preProcessVideo.image_tensor.dim_color)) # batch data that will pass forward
        
        batch_labels = np.zeros((batch_size,2)) # batch labels that will pass forward
        
        #############################################################
        # Here is how the batch data is split by callback
        if(((batch+1)*batch_size) <= self.train_metadata.shape[0]):
            train_metadata_ = self.train_metadata.iloc[
                batch*batch_size:(batch+1)*batch_size,
                :
            ]
        else:
            train_metadata_ = self.train_metadata.iloc[
                batch*batch_size:,
                :
            ]
        
        #############################################################
        video_posi = 0
        name_list = train_metadata_['name'].to_list()
        label_list = train_metadata_["label"].to_list()
        
        for name,label in zip(name_list,label_list):
            path_ = f"{self.preProcessVideo.image_tensor.path}/{name}"
            batch_data[video_posi] = self.preProcessVideo.pre_process_video(path_)
            
            if(label == "FAKE"):
                batch_labels[video_posi][0] = 1
            else:
                batch_labels[video_posi][1] = 1
                
            video_posi += 1
                
        return batch_data, batch_labels

    def generator(self):
        batch_size = self.preProcessVideo.image_tensor.batch_size
        while True:
            if(len(self.train_metadata["name"])%batch_size == 0):
                num_batches = int(len(self.train_metadata["name"])/batch_size)
            else:
                num_batches = int(len(self.train_metadata["name"])/batch_size) + 1
            
            for batch in range(num_batches): # we iterate over the number of batches
                yield self.getBatchData(batch)


class Models():
    def __init__(self,image_tensor:ImageTensor,layersAux:LayersAux) -> None:
        self.image_tensor = image_tensor
        self.layersAux = layersAux

    def create_checkpoint(self,name:str) -> list:
        curr_dt_time = datetime.datetime.now()

        model_name = name + '_' + str(curr_dt_time).replace(' ','').replace(':','_') + '/'
            
        if(not os.path.exists(model_name)):
            os.mkdir(model_name)

        filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        
        LR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1, patience=4)

        return [checkpoint, LR]

    def defineModel(self) -> Sequential :
        inputShape = (
            len(self.image_tensor.img_idx),
            self.image_tensor.img_size,
            self.image_tensor.img_size,
            self.image_tensor.dim_color
            )
        
        print(inputShape)

        model = Sequential([
            Conv3D(16, self.layersAux.make3dFilter(5), activation='relu', input_shape=inputShape),
            MaxPooling3D( self.layersAux.make3dFilter(2), padding='same'),
            BatchNormalization(),

            Conv3D(32,  self.layersAux.make3dFilter(3), activation='relu'),
            MaxPooling3D(pool_size=(1,2,2), padding='same'),
            BatchNormalization(),

            Conv3D(64,  self.layersAux.make3dFilter(3), activation='relu'),
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

        model.compile(
            optimizer=optimizers.Adam(),
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
            )

        return model


class SavePlots():
    def __init__(self,image_tensor:ImageTensor,name_save:str) -> None:
        self.image_tensor = image_tensor
        self.name_save = name_save + '_' + re.sub(r"\.[0-9]+","",str(datetime.datetime.now())).replace(":","-").replace(" ","_")

    def plotModelHistory(self,h:Sequential)->None:
        path = self.image_tensor.path_plots

        if(not os.path.exists(path)):
            os.mkdir(path)

        try: 
            fig, ax = plt.subplots(1, 2, figsize=(15,4))
            ax[0].plot(h.history['loss'])   
            ax[0].plot(h.history['val_loss'])
            ax[0].legend(['loss','val_loss'])
            ax[0].title.set_text("Train loss vs Validation loss")

            ax[1].plot(h.history['categorical_accuracy'])   
            ax[1].plot(h.history['val_categorical_accuracy'])
            ax[1].legend(['categorical_accuracy','val_categorical_accuracy'])
            ax[1].title.set_text("Train accuracy vs Validation accuracy")
            
            
            plt.savefig(f"{path}/{self.name_save}")

            with open(f'{path}/{self.name_save}.json', 'w+', encoding='utf-8') as f:
                json.dump({
                    "categorical_accuracy": float(max(h.history['categorical_accuracy'])),
                    "val_categorical_accuracy": float(max(h.history['val_categorical_accuracy']))
                },
                f,
                ensure_ascii=False,
                indent=4)

                f.close()
        except Exception as e:
            print(f"Erro ao salvar graficos: {e}")

    def plotAtributos(self)->None:
        path = self.image_tensor.path_plots

        if(not os.path.exists(path)):
            os.mkdir(path)

        try:
            with open(f'{path}/{self.name_save}.json', 'w+', encoding='utf-8') as f:
                    json.dump(self.image_tensor.get_atributos(),
                    f,
                    ensure_ascii=False,
                    indent=4)

                    f.close()
        except Exception as e:
            print(f"Erro ao salvar atributos: {e}")


