import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from tensorflow import keras
from train_facade import *
from tqdm import tqdm

class MakeTest():
    def __init__(self,pre_process:PreProcessVideo,save_plots:SavePlots,test_metadata:pd.DataFrame) -> None:
        self.pre_process = pre_process
        self.save_plots = save_plots
        self.test_metadata = test_metadata
        

    def teste_1(self,path_checkpoint:str,path_data_test:str):
        model = keras.models.load_model(path_checkpoint)

        predict = []
        gabarito = list(self.test_metadata["label"])
        for filename_ in tqdm(self.test_metadata["name"],total=self.test_metadata.shape[0]):
            vid = self.pre_process.pre_process_video(f"{path_data_test}/{filename_}")
            vid = np.array([vid])
            predict.append("FAKE" if model.predict(vid).argmax() == 0 else "REAL")

        result = {
            "acc"       : accuracy_score(gabarito,predict),
            "rec"       : recall_score(gabarito,predict,pos_label="FAKE"),
            "f1"        : f1_score(gabarito,predict,pos_label="FAKE"),
            "conf_mat"  : confusion_matrix(gabarito,predict,labels=["FAKE", "REAL"]).tolist()
        }

        print(json.dumps(result,indent=1))

        path = self.pre_process.image_tensor.path_plots

        if(not os.path.exists(path)):
            os.mkdir(path)

        name_save = "MODELO_TESTE_49_2022-04-2916_58_51.339119"
        try:
            with open(f'{path}/{name_save}.json', 'w+', encoding='utf-8') as f:
                    json.dump(result,
                    f,
                    ensure_ascii=False,
                    indent=4)

                    f.close()
        except Exception as e:
            print(f"Erro ao salvar atributos: {e}")
        