from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
import cv2
import os

class DataGenerator(Sequence):
    def __init__(
        self,
        metadata:pd.DataFrame,
        len_frames:int,
        frames:int,
        batch_size:int,
        dim:tuple,
        n_channels:int,
        n_classes:int,
        shuffle:bool,
        path:str
        ):
        self.metadata = metadata
        self.len_frames = len_frames
        self.frames = frames
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.path = path
        #self.img_idx = np.round(np.linspace(
        #    (self.len_frames//2) - (self.frames//2),
        #    (self.len_frames//2) + (self.frames//2),
        #    self.frames)
        #    ).astype(int)
        self.img_idx = np.round(np.linspace(0, self.len_frames, self.frames)).astype(int)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.metadata.shape[0] / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.metadata.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of ids in DataFrame
        list_metadata_temp = self.metadata.iloc[indexes]

        # Generate data
        X, y = self.__data_generation(list_metadata_temp)

        return X, y

    def __data_generation(self, list_metadata_temp:pd.DataFrame):
        'Generates data containing batch_size samples' # X : (n_samples, frames,*dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size,  self.frames ,*self.dim, self.n_channels),dtype=np.uint8)
        y = np.zeros((self.batch_size,self.n_classes), dtype=np.float32)

        def pre_process_video(path:str):
            if(not os.path.exists(path)):
                raise Exception(f"Arquivo {path} nao encontrado")

            frames = []
            vidcapture = cv2.VideoCapture(path)
            index = 0
            j = 0
            while(vidcapture.isOpened()):
                rent, frame = vidcapture.read()
                if(not rent):
                    break
                else:
                    if(len(self.img_idx) - 1 < j):
                        break
                    else:
                        if(index == self.img_idx[j]):
                            if(self.n_channels == 1):
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = cv2.resize(frame,self.dim)
                            frames.append(frame)
                            j += 1

                        index += 1

            vidcapture.release()
            cv2.destroyAllWindows()
            
            if(self.n_channels == 1):
                frames = np.array(frames)
                return frames.reshape(frames.shape[0],frames.shape[1],frames.shape[2],self.n_channels)
                
            
            elif(self.n_channels == 3):
                return np.array(frames) 


        # Generate data
        names = list_metadata_temp['name'].to_list()
        labels = list_metadata_temp['label'].to_list()

        for i in range(list_metadata_temp.shape[0]):
            # Store sample
            X[i,] = pre_process_video(f"{self.path}/{names[i]}")

            if(labels[i] == "FAKE"):
                y[i][0] = 1
            else:
                y[i][1] = 1

        return X, y