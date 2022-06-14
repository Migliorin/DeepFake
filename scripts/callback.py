from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
import os


class Callback():
    def __init__(self):
        pass
    def create_checkpoint(self,path_checkpoint:str,name:str)->list:
        curr_dt_time = datetime.now()

        if(not os.path.exists(path_checkpoint)):
            raise Exception(f"Caminho para salvar checkpoint nao existe: {path_checkpoint}")

        model_name= f"{path_checkpoint}/{name}_{str(curr_dt_time).replace(' ','').replace(':','_')}/"
            
        if(not os.path.exists(model_name)):
            os.mkdir(model_name)

        filepath = model_name + 'model-{epoch:05d}-{loss:.5f}-{categorical_accuracy:.5f}-{val_loss:.5f}-{val_categorical_accuracy:.5f}.h5'

        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    
        reduce_lr = ReduceLROnPlateau(patience=10)

        return [checkpoint, reduce_lr]