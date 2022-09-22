import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"

import tensorflow as tf
from model import DenseNet
from tensorflow.keras.datasets import cifar10
import numpy as np

# parameter
batch_size = 256
epochs = 1
ratio1 = 0.7
ratio2 = 0.15


def split(dataset, label, ratio1, ratio2):
    [x_train, x_validation, x_test] = np.split(dataset, [int(len(dataset) * ratio1), int(len(dataset) * ratio2)])
    [y_train, y_validation, y_test] = np.split(label, [int(len(label) * ratio1), int(len(label) * ratio2)])

    # DenseNet
    x_train = tf.keras.applications.densenet.preprocess_input(x_train)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    x_validation = tf.keras.applications.densenet.preprocess_input(x_validation)
    y_validation = tf.keras.utils.to_categorical(y_validation, 10)
    x_test = tf.keras.applications.densenet.preprocess_input(x_test)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)


def main():
    model = DenseNet().build()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # blime : To be fast, the dataset is shrunk by a factor of SCALE
    SCALE = 1
    x = x[:len(x)//SCALE]
    y = y[:len(y)//SCALE]

    (x_train, y_train), (x_validation, y_validation), (x_test, y_test) = split(x, y, ratio1, ratio2)


    tf.config.optimizer.set_experimental_options({
        'layout_optimizer': False,
    })
    # tf.profiler.experimental.start('nv-logdir-825')
    print(tf.config.optimizer.get_experimental_options())
    log_dir="dl-matmul-001"
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                             profile_batch='10, 15')
    model.fit(x_train, y_train,
              epochs=epochs,
              verbose=1,
              validation_data=(x_validation, y_validation)
              ,
              callbacks=[tb_callback])
    # tf.profiler.experimental.stop()

    # score = model.evaluate(x_test, y_test)
    # print('Test loss of model:', score[0])
    # print('Test accuracy of model:', score[1])


if __name__ == "__main__":
    main()
