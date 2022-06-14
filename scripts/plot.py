from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
from datetime import datetime
import re

class SaveImage():
    def __init__(self):
        pass
    def save_model_history(self,model:History, **kwargs):
        """
            name_save: O nome que sera salvo o grafico

            metric: Metrica usada

            path_save: Caminho para salvar
        
        """
        name_save = kwargs["name_save"][::]

        date = re.sub(r"\.[0-9]+","",str(datetime.now())).replace(":","-").replace(" ","_")
        name_save = f'{name_save}_{date}'
        
        h = model

        fig, ax = plt.subplots(1, 2, figsize=(15,4))
        ax[0].plot(h.history['loss'])   
        ax[0].plot(h.history['val_loss'])
        ax[0].legend(['loss','val_loss'])
        ax[0].title.set_text("Train loss vs Validation loss")

        metrica = kwargs['metric']
        metrica_val = f"val_{metrica}"
        
        text = metrica.replace('_',' ')
        text = " ".join([x.capitalize() for x in text.split(' ')])

        ax[1].plot(h.history[metrica])   
        ax[1].plot(h.history[metrica_val])

        ax[1].legend([metrica,metrica_val])
        ax[1].title.set_text(f"Train {text} vs Validation {text}")

        plt.savefig(f"{kwargs['path_save']}/{name_save}")