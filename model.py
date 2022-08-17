from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Flatten
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class DenseNet():
    def __init__(self):
        pass

    def build(self):
        input_tensor = tf.keras.Input(shape=(32,32,3)) # cifar10
        resized_tensor = tf.keras.layers.Lambda(lambda image : tf.image.resize(image, (224, 224)))(input_tensor)
        base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=resized_tensor, input_shape=(32,32,3), pooling='max', classes=1000)

        for layer in base_model.layers:
            layer.trainable = False

        output = base_model.output
        flatten_output = Flatten()(output)

        def fc(num_classes, _input, activation, trainable):
            x = _input
            x = Dense(512, kernel_regularizer=l2(0.001), trainable=trainable)(x)
            x = BatchNormalization(trainable=trainable)(x)
            x = Activation('relu', trainable=trainable)(x)
            x = Dropout(0.2)(x)

            x = Dense(512, kernel_regularizer=l2(0.001), trainable=trainable)(x)
            x = BatchNormalization(trainable=trainable)(x)
            x = Activation('relu', trainable=trainable)(x)
            x = Dropout(0.2)(x)
            return Dense(num_classes, activation=activation, trainable=trainable)(x)

        prediction = fc(10, flatten_output, 'softmax', True)

        model = Model(inputs=base_model.input, outputs=prediction)

        model.summary()

        model.compile(  loss="categorical_crossentropy",
                        optimizer='adam',
                        metrics=['accuracy'])

        return model