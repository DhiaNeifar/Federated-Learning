from __init__ import NUMBER_OF_CLIENTS

import os
import tensorflow as tf
from tensorflow.keras import layers, models


def split_dataset(dataset, dataset_length):

    train_size = int(dataset_length * .7)
    val_size = int(dataset_length * .2)
    test_size = dataset_length - train_size - val_size
    return dataset.take(train_size), dataset.skip(train_size).take(val_size), dataset.skip(train_size + val_size).take(test_size)


def preprocess(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) 
    image = image / 255.0
    label = tf.strings.to_number(tf.strings.split(label, ' ')[-1], out_type=tf.int32)
    return image, label


def load_dataset(data_path):
    files = [os.path.join(data_path, f'{i}.png') for i in range(len(os.listdir(data_path)) - 1)]
    X = tf.data.Dataset.from_tensor_slices(files)
    y = tf.data.TextLineDataset(os.path.join(data_path, 'output.txt'))
    dataset = tf.data.Dataset.zip((X, y))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = dataset.map(preprocess,num_parallel_calls=AUTOTUNE)
    return dataset


def load_local_data(client):
    print(client)
    path = f'/home/neifar/federated dataset/{NUMBER_OF_CLIENTS} clients/client {client + 1}'
    dict_dataset = {
        'train': load_dataset(os.path.join(path, 'train')),
        'val': load_dataset(os.path.join(path, 'val')),
    }
    return dict_dataset


def create_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(256, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(512, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def baseline_model(input_shape, model_name):
    if model_name == 'proposed':
        return create_model(input_shape)
    base_model = None
    if model_name == 'resnet50':
        base_model = tf.keras.applications.resnet50.ResNet50(weights=None, include_top=False, input_shape=input_shape)
    if model_name == 'densenet':
        base_model = tf.keras.applications.DenseNet121(weights=None, include_top=False, input_shape=input_shape)
    if model_name == 'mobilenet':
        base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=input_shape)
    if model_name == 'nasnet':
        base_model = tf.keras.applications.NASNetMobile(weights=None, include_top=False, input_shape=input_shape)    
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    return model

def find_client_index():
    index = 1
    for file in os.listdir():
        if file[0] == 'c' and file[-1] == 'g':
            index += 1
    return index
