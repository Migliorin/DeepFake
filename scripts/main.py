from train_facade import *
from test_facade import *

class MainClass():
    def main_teste_1(self):
        image_tensor = ImageTensor()
        save_plots = SavePlots(image_tensor)

        BATCH_SIZE = image_tensor.batch_size
        EPOCHS = image_tensor.epochs
        PATH = image_tensor.path
        NAME = "MODELO_TESTE_1"
        
        
        train_metadata = pd.read_json(f"{PATH}/metadata.json")
        train_metadata = train_metadata.T
        train_metadata.reset_index(inplace=True)
        train_metadata.rename({"index":"name"},axis=1,inplace=True)

        

        pre_process_video = PreProcessVideo(image_tensor)
        
        layers_aux = LayersAux()
        models = Models(image_tensor,layers_aux)
        
        model = models.defineModel()
        callbacks_list = models.create_checkpoint(NAME)

        train, test = train_test_split(
            train_metadata,
            test_size=0.33,
            random_state=42,
            stratify=train_metadata["label"]
            )

        train_generator = BatchData(train,pre_process_video).generator()
        val_generator = BatchData(test,pre_process_video).generator()

        if (train.shape[0]%BATCH_SIZE) == 0:
            steps_per_epoch = int(train.shape[0]/BATCH_SIZE)
        else:
            steps_per_epoch = (train.shape[0]//BATCH_SIZE) + 1

        if (test.shape[0]%BATCH_SIZE) == 0:
            validation_steps = int(test.shape[0]/BATCH_SIZE)
        else:
            validation_steps = (test.shape[0]//BATCH_SIZE) + 1

        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks_list,
            validation_data=val_generator,
            validation_steps=validation_steps,
            class_weight=None,
            workers=1,
            initial_epoch=0
            )

        save_plots.plotModelHistory(model.history,NAME)
        save_plots.plotAtributos(NAME)


    def main_train_49(self):
        image_tensor = ImageTensor()
        image_tensor.switch_image_tensor_49()


        BATCH_SIZE = image_tensor.batch_size
        EPOCHS = image_tensor.epochs
        PATH = image_tensor.path
        NAME = "MODELO_TESTE_49"
        
        save_plots = SavePlots(image_tensor,NAME)
        
        train_metadata = pd.read_csv(f"{PATH}/metadata.csv")

        pre_process_video = PreProcessVideo(image_tensor)
        
        layers_aux = LayersAux()
        models = Models(image_tensor,layers_aux)
        
        model = models.defineModel()
        callbacks_list = models.create_checkpoint(NAME)

        train, test = train_test_split(
            train_metadata,
            test_size=0.33,
            random_state=42,
            stratify=train_metadata["label"]
            )

        train_generator = BatchData(train,pre_process_video).generator()
        val_generator = BatchData(test,pre_process_video).generator()

        if (train.shape[0]%BATCH_SIZE) == 0:
            steps_per_epoch = int(train.shape[0]/BATCH_SIZE)
        else:
            steps_per_epoch = (train.shape[0]//BATCH_SIZE) + 1

        if (test.shape[0]%BATCH_SIZE) == 0:
            validation_steps = int(test.shape[0]/BATCH_SIZE)
        else:
            validation_steps = (test.shape[0]//BATCH_SIZE) + 1

        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks_list,
            validation_data=val_generator,
            validation_steps=validation_steps,
            class_weight=None,
            workers=1,
            initial_epoch=0
            )

        save_plots.plotModelHistory(model.history)
        save_plots.plotAtributos()

    def main_train_49_01(self):
        image_tensor = ImageTensor()
        image_tensor.switch_image_tensor_49()


        BATCH_SIZE = image_tensor.batch_size
        EPOCHS = image_tensor.epochs
        PATH = image_tensor.path
        NAME = "MODELO_TESTE_49"
        
        save_plots = SavePlots(image_tensor,NAME)
        
        train_metadata = pd.read_csv(f"{PATH}/metadata_01.csv")

        pre_process_video = PreProcessVideo(image_tensor)
        
        layers_aux = LayersAux()
        models = Models(image_tensor,layers_aux)
        
        model = models.defineModel()
        callbacks_list = models.create_checkpoint(NAME)

        train, test = train_test_split(
            train_metadata,
            test_size=0.33,
            random_state=42,
            stratify=train_metadata["label"]
            )

        train_generator = BatchData(train,pre_process_video).generator()
        val_generator = BatchData(test,pre_process_video).generator()

        if (train.shape[0]%BATCH_SIZE) == 0:
            steps_per_epoch = int(train.shape[0]/BATCH_SIZE)
        else:
            steps_per_epoch = (train.shape[0]//BATCH_SIZE) + 1

        if (test.shape[0]%BATCH_SIZE) == 0:
            validation_steps = int(test.shape[0]/BATCH_SIZE)
        else:
            validation_steps = (test.shape[0]//BATCH_SIZE) + 1

        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks_list,
            validation_data=val_generator,
            validation_steps=validation_steps,
            class_weight=None,
            workers=1,
            initial_epoch=0
            )

        save_plots.plotModelHistory(model.history)
        save_plots.plotAtributos()


    def main_train_49_02(self):
        image_tensor = ImageTensor()
        image_tensor.switch_image_tensor_49_01()


        BATCH_SIZE = image_tensor.batch_size
        EPOCHS = image_tensor.epochs
        PATH = image_tensor.path
        NAME = "MODELO_TESTE_49"
        
        save_plots = SavePlots(image_tensor,NAME)
        
        train_metadata = pd.read_csv(f"{PATH}/metadata_01.csv")

        pre_process_video = PreProcessVideo(image_tensor)
        
        layers_aux = LayersAux()
        models = Models(image_tensor,layers_aux)
        
        model = models.defineModel()
        callbacks_list = models.create_checkpoint(NAME)

        train, test = train_test_split(
            train_metadata,
            test_size=0.33,
            random_state=42,
            stratify=train_metadata["label"]
            )

        train_generator = BatchData(train,pre_process_video).generator()
        val_generator = BatchData(test,pre_process_video).generator()

        if (train.shape[0]%BATCH_SIZE) == 0:
            steps_per_epoch = int(train.shape[0]/BATCH_SIZE)
        else:
            steps_per_epoch = (train.shape[0]//BATCH_SIZE) + 1

        if (test.shape[0]%BATCH_SIZE) == 0:
            validation_steps = int(test.shape[0]/BATCH_SIZE)
        else:
            validation_steps = (test.shape[0]//BATCH_SIZE) + 1

        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks_list,
            validation_data=val_generator,
            validation_steps=validation_steps,
            class_weight=None,
            workers=1,
            initial_epoch=0
            )

        save_plots.plotModelHistory(model.history)
        save_plots.plotAtributos()


    def main_train_49_03(self)->None:

        def train_val_test(train_metadata:pd.DataFrame)->list:

            fake = train_metadata[train_metadata["label"] == "FAKE"]

            real = train_metadata[train_metadata["label"] == "REAL"]

            train_fake, test_fake = train_test_split(
                        fake,
                        test_size=0.3,
                        random_state=42
                        )

            train_real, test_real = train_test_split(
                        real,
                        test_size=0.3,
                        random_state=42
                        )

            train = pd.concat([train_fake,train_real])

            val_test = pd.concat([test_fake,test_real])

            val, test = train_test_split(
                        val_test,
                        test_size=0.5,
                        random_state=42
                        )

            return train,val,test
        
        image_tensor = ImageTensor()
        image_tensor.switch_image_tensor_49_02()

        BATCH_SIZE = image_tensor.batch_size
        EPOCHS = image_tensor.epochs
        PATH = image_tensor.path
        NAME = "CONV_LSTM_01"

        save_plots = SavePlots(image_tensor,NAME)
        train_metadata = pd.read_csv(f"{PATH}/metadata_02.csv")

        layers_aux = LayersAux()
        models = Models(image_tensor,layers_aux)
        
        pre_process_video = PreProcessVideo(image_tensor)

        model = models.InceptionV3_artigo()
        callbacks_list = models.create_checkpoint_02(NAME)

        train, val, test = train_val_test(train_metadata)

        train_generator = BatchData(train,pre_process_video).generatorWithColor()
        val_generator = BatchData(val,pre_process_video).generatorWithColor()

        if (train.shape[0]%BATCH_SIZE) == 0:
            steps_per_epoch = int(train.shape[0]/BATCH_SIZE)
        else:
            steps_per_epoch = (train.shape[0]//BATCH_SIZE) + 1

        if (test.shape[0]%BATCH_SIZE) == 0:
            validation_steps = int(test.shape[0]/BATCH_SIZE)
        else:
            validation_steps = (test.shape[0]//BATCH_SIZE) + 1

        model.fit_generator(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks_list,
            validation_data=val_generator,
            validation_steps=validation_steps,
            class_weight=None,
            workers=1,
            initial_epoch=0
            )

        save_plots.plotModelHistory_02(model.history)
        save_plots.plotAtributos()


    def main_test_48(self):
        image_tensor = ImageTensor()
        image_tensor.switch_image_tensor_49()

        NAME = "MODELO_TESTE_49"
        save_plots = SavePlots(image_tensor,NAME)
        pre_process_video = PreProcessVideo(image_tensor)

        test_metadata = pd.read_json(f"../datasets/dfdc_train_part_48/metadata.json")
        test_metadata = test_metadata.T
        test_metadata.reset_index(inplace=True)
        test_metadata.rename({"index":"name"},axis=1,inplace=True)

        test = MakeTest(pre_process_video,save_plots,test_metadata)
        test.teste_1(
            "./MODELO_TESTE_49_2022-04-2916_58_51.339119/model-00009-0.62787-0.65643-0.78289-0.58772.h5",
            "../datasets/dfdc_train_part_48"
            )

    def main_test_48_01(self):
        image_tensor = ImageTensor()
        image_tensor.switch_image_tensor_49()

        NAME = "MODELO_TESTE_49"
        save_plots = SavePlots(image_tensor,NAME)
        pre_process_video = PreProcessVideo(image_tensor)

        test_metadata = pd.read_json(f"../datasets/dfdc_train_part_48/metadata.json")
        test_metadata = test_metadata.T
        test_metadata.reset_index(inplace=True)
        test_metadata.rename({"index":"name"},axis=1,inplace=True)

        test = MakeTest(pre_process_video,save_plots,test_metadata)
        test.teste_2(
            "./MODELO_TESTE_49_2022-05-0316_10_45.606091/model-00010-0.58094-0.71925-0.63546-0.68254.h5",
            "../datasets/dfdc_train_part_48"
            )

if __name__ == "__main__":

    main_ = MainClass()
    # main_.main_train_49_01()
    main_.main_train_49_03() # 19/05/2022

