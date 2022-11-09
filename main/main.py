import random
import os
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model

from utils.dateset_gen import *
from utils.model_create import *
from utils.evaluate import plot_metric


def train_lstm():
    path = "/home/kushojha/Action-Detection/RWF-2000/train/FightKeyPoints/"
    X_train, y_train = [], []
    for folder in os.listdir(path):
        window = []
        length = len(os.listdir(path+folder))
        sample = random.sample(range(0, length), k=60)
        sample.sort()
        for i in sample:
            keypoints = np.load(
                f"/home/kushojha/Action-Detection/RWF-2000/train/FightKeyPoints/{folder}/{i}.npy")
            res = keypoints.flatten()
            window.append(res)
        X_train.append(window)
        y_train.append(1)
    path = "/home/kushojha/Action-Detection/RWF-2000/train/NonFightKeyPoints/"
    for folder in os.listdir(path):
        window = []
        length = len(os.listdir(path+folder))
        sample = random.sample(range(0, length), k=60)
        sample.sort()
        for i in sample:
            keypoints = np.load(
                f"/home/kushojha/Action-Detection/RWF-2000/train/NonFightKeyPoints/{folder}/{i}.npy")
            res = keypoints.flatten()
            window.append(res)
        X_train.append(window)
        y_train.append(0)

    path = "/home/kushojha/Action-Detection/RWF-2000/val/FightKeyPoints/"
    X_test, y_test = [], []
    for folder in os.listdir(path):
        window = []
        length = len(os.listdir(path+folder))
        sample = random.sample(range(0, length), k=60)
        sample.sort()
        for i in sample:
            keypoints = np.load(
                f"/home/kushojha/Action-Detection/RWF-2000/val/FightKeyPoints/{folder}/{i}.npy")
            res = keypoints.flatten()
            window.append(res)
        X_test.append(window)
        y_test.append(1)
    path = "/home/kushojha/Action-Detection/RWF-2000/val/NonFightKeyPoints/"
    for folder in os.listdir(path):
        window = []
        length = len(os.listdir(path+folder))
        sample = random.sample(range(0, length), k=60)
        sample.sort()
        for i in sample:
            keypoints = np.load(
                f"/home/kushojha/Action-Detection/RWF-2000/val/NonFightKeyPoints/{folder}/{i}.npy")
            res = keypoints.flatten()
            window.append(res)
        X_test.append(window)
        y_test.append(0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model = create_lstm()

    plot_model(model, to_file='lstm_model_structure_plot.png',
               show_shapes=True, show_layer_names=True)

    model.summary()

    early_stopping = EarlyStopping(
        min_delta=0.001,
        patience=50,
        restore_best_weights=True,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(epsilon=0.01),
        loss='binary_crossentropy',
        metrics=['binary_accuracy']
    )

    history = model.fit(
        X_train, y_train,
        batch_size=32,
        validation_data=(X_test, y_test),
        epochs=200,
        verbose=1,
        callbacks=[early_stopping]
    )

    plot_metric(history, 'loss', 'val_loss',
                'Total Loss vs Total Validation Loss')
    plot_metric(history, 'binary_accuracy', 'val_binary_accuracy',
                'Total Accuracy vs Total Validation Accuracy')

    model.save()


def train_convlstm():

    features, labels, video_files_paths = create_dataset()
    model = create_convlstm_model()
    
    plot_model(model, to_file = 'convlstm_model_structure_plot.png', show_shapes = True, show_layer_names = True)

    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            mode='min',
                                            restore_best_weights=True)

    model.compile(loss='binary_crossentropy', optimizer='Adam',
                  metrics=["binary_accuracy"])

    history = model.fit(x=features, y=labels,
                        epochs=50,
                        batch_size=4,
                        shuffle=True,
                        validation_split=0.2,
                        callbacks=[early_stopping_callback])

    plot_metric(history, 'loss', 'val_loss',
                'Total Loss vs Total Validation Loss')
    plot_metric(history, 'binary_accuracy', 'val_binary_accuracy',
                'Total Accuracy vs Total Validation Accuracy')
   
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    
    model_file_name = f'action_detect_{current_date_time_string}_convlstm'
    model.save(model_file_name)
    

def train_lrcn():

    
    features, labels, video_files_paths = create_dataset()
    model = create_LRCN_model()
    
    plot_model(model, to_file = 'lrcn_model_structure_plot.png', show_shapes = True, show_layer_names = True)

    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            patience=10,
                                            mode='min',
                                            restore_best_weights=True)

    model.compile(loss='binary_crossentropy', optimizer='Adam',
                  metrics=["binary_accuracy"])

    history = model.fit(x=features, y=labels,
                        epochs=50,
                        batch_size=4,
                        shuffle=True,
                        validation_split=0.2,
                        callbacks=[early_stopping_callback])

    plot_metric(history, 'loss', 'val_loss',
                'Total Loss vs Total Validation Loss')
    plot_metric(history, 'binary_accuracy', 'val_binary_accuracy',
                'Total Accuracy vs Total Validation Accuracy')
   
    date_time_format = '%Y_%m_%d__%H_%M_%S'
    current_date_time_dt = dt.datetime.now()
    current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)
    
    model_file_name = f'action_detect_{current_date_time_string}_lrcn'
    model.save(model_file_name)