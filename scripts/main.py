import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import sys

sys.setrecursionlimit(5000)

from generator import *
from callback import *
from model import *
from plot import *

from sklearn.model_selection import train_test_split

NAME_LOGIN = os.getlogin()

PATH = f"/home/{NAME_LOGIN}/dfdc/datasets/dfdc_train_part_49"
METADATA_PATH = f"{PATH}/metadata_02.csv"
CHECKPOINT_PATH = f"/home/{NAME_LOGIN}/dfdc/models"
PLOTS_PATH = f"/home/{NAME_LOGIN}/dfdc/plots"

FRAMES = 20

NAME_MODEL = f"Efficient_{FRAMES}_frames"
METRIC = "categorical_accuracy"
EPOCHS = 64

params = {
    "len_frames"   :290,
    "frames"       :FRAMES,
    "batch_size"   :10,
    "dim"          :(480,480),
    "n_channels"   :3,
    "n_classes"    :2,
    "shuffle"      :True,
    "path"         :PATH
    
}
simbol = '#'


print(f"{simbol*5} Obtendo dataframe geral {simbol*5}")
metadata = pd.read_csv(METADATA_PATH)
train, test = train_test_split(metadata,test_size=0.30,random_state=42,shuffle=True)
val, test = train_test_split(test,test_size=0.50,random_state=42,shuffle=True)
print(f"{simbol*5} Concluido {simbol*5}\n")

print(f"{simbol*5} Criando Generators {simbol*5}")
train_generator = DataGenerator(metadata=train,**params)
val_generator = DataGenerator(metadata=val,**params)
print(f"{simbol*5} Concluido {simbol*5}\n")


print(f"{simbol*5} Iniciando modelo {simbol*5}")
model = Models()
model = model.EfficientNetV2L(inputShape=(
    *params["dim"],
    params["n_channels"]
    )
)

callback = Callback()
callback = callback.create_checkpoint(CHECKPOINT_PATH,NAME_MODEL)
print(f"{simbol*5} Concluido {simbol*5}\n")


print(f"{simbol*5} Iniciando Treinamento {simbol*5}")
model.fit(
    train_generator,
    validation_data=val_generator,
    use_multiprocessing=True,
    workers=6,
    verbose=1,
    epochs=EPOCHS,
    callbacks=callback
)
print(f"{simbol*5} Concluido {simbol*5}\n")


print(f"{simbol*5} Salvando Graficos {simbol*5}")
paramns_save = {
    "name_save": NAME_MODEL,
    "metric": METRIC,
    "path_save": PLOTS_PATH
}
plot_img = SaveImage()
plot_img.save_model_history(model.history,**paramns_save)

print(f"{simbol*5} Concluido {simbol*5}\n")