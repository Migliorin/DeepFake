import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from model import *
import cv2


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import numpy as np

from tqdm import tqdm
import json

import matplotlib.pyplot as plt

NAME_LOGIN = os.getlogin()
PATH = f"/home/{NAME_LOGIN}/dfdc"

MODEL_PATH = f"{PATH}/models/CONV_LSTM_80_frames_2022-05-2620_34_20.894420/model-00032-0.49215-0.75952-0.84151-0.62222.h5"
METADATA_PATH = f"{PATH}/datasets/dfdc_train_part_49"
METADATA_NAME = "metadata_02.csv"


model = Models()
model = model.InceptionV3_artigo_v02(inputShape=(299,299,3))

model.load_weights(MODEL_PATH)

metadata = pd.read_csv(f"{METADATA_PATH}/{METADATA_NAME}")

train, test = train_test_split(metadata,test_size=0.30,random_state=42,shuffle=True)
val, test = train_test_split(test,test_size=0.50,random_state=42,shuffle=True)

name = list(test['name'])
label = list(test['label'])

predict = []
for name_ in tqdm(name,total=len(name)):
    frames = []
    path = f"{METADATA_PATH}/{name_}"
    vidcapture = cv2.VideoCapture(path)
    while(vidcapture.isOpened()):
        rent, frame = vidcapture.read()
        if(not rent):
            break
        else:
            frame = cv2.resize(frame,(299,299))
            frames.append(frame)

    frames = np.array([frames])
    predict.append("FAKE" if model.predict(frames).argmax() == 0 else "REAL")

disp = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix(label,predict,labels=["FAKE","REAL"]),
    display_labels=["FAKE","REAL"]
)

disp.plot()

FOLDER = "/"+"/".join(MODEL_PATH.split("/")[:-1])

plt.savefig(f'{FOLDER}/confusion_matrix')

try:
    with open(f'{FOLDER}/metricas.json', 'w+', encoding='utf-8') as f:
            json.dump({
                "acuracia": accuracy_score(label,predict),
                "recall_fake": recall_score(label,predict,pos_label="FAKE"),
                "reacll_real": recall_score(label,predict,pos_label="REAL"),
                "f1_score": f1_score(label,predict,pos_label="FAKE")
            },
            f,
            ensure_ascii=False,
            indent=4)

            f.close()
except Exception as e:
    print(f"Erro ao salvar atributos: {e}")

