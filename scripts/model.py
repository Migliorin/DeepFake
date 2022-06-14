from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed, Lambda, Input
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers, Model, Sequential
import tensorflow_hub as hub


class Models():
    def __init__(self):
        pass

    def InceptionV3_artigo_v01(self,inputShape:tuple) -> Sequential:
        
        model = InceptionV3()

        modelIv3 = Sequential([
            TimeDistributed(Lambda(preprocess_input),input_shape=inputShape),
            TimeDistributed(model)
        ], name="inception_lstm")

        for layer in modelIv3.layers:
            layer.trainable = False
            
        modelIv3.trainable = False

            
        modelIv3.add(LSTM(2048,dropout=0.5))
        modelIv3.add(Dense(512))
        modelIv3.add(Dropout(0.5))
        modelIv3.add(Dense(2, activation='softmax'))
        
        OPTIMIZER = optimizers.Adam(learning_rate=1e-5, decay=1e-6)
        modelIv3.compile(optimizer=OPTIMIZER,metrics=['categorical_accuracy'],loss='categorical_crossentropy')

        return modelIv3

    def InceptionV3_artigo_v02(self,inputShape:tuple) -> Sequential:
        
        image = Input(shape=(None,*inputShape),name='image_input')
        cnn = InceptionV3()

        x = preprocess_input(image)

        cnn.trainable = False
        encoded_frame = TimeDistributed(Lambda(lambda x: cnn(x)))(x)

        encoded_vid = LSTM(2048,dropout=0.5)(encoded_frame)
        layer1 = Dense(512)(encoded_vid)
        dropout1 = Dropout(0.5)(layer1)
        outputs = Dense(2, activation='softmax')(dropout1)

        model = Model(inputs=[image],outputs=outputs)
        
        OPTIMIZER = optimizers.Adam(learning_rate=1e-5, decay=1e-6)
        model.compile(optimizer=OPTIMIZER,metrics=['categorical_accuracy'],loss='categorical_crossentropy')

        return model

    def EfficientNetV2L(self, inputShape:tuple) -> Sequential:
        cnn = Sequential([
        hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/classification/2")
            ])
        cnn.build([None, *inputShape])  # Batch input shape max.
        image = Input(shape=(None,*inputShape),name='image_input')
        cnn.trainable = False
        encoded_frame = TimeDistributed(Lambda(lambda x: cnn(x)))(image)
        
        encoded_vid = LSTM(2048,dropout=0.5)(encoded_frame)
        layer1 = Dense(512)(encoded_vid)
        dropout1 = Dropout(0.5)(layer1)
        outputs = Dense(2, activation='softmax')(dropout1)

        model = Model(inputs=[image],outputs=outputs)

        OPTIMIZER = optimizers.Adam(learning_rate=1e-5, decay=1e-6)
        model.compile(optimizer=OPTIMIZER,metrics=['categorical_accuracy'],loss='categorical_crossentropy')
        return model