from config import *

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

def get_train_labels():
    df = pd.read_csv(DATASET_DIR+TRAIN_LABELS, header = None)
    return df

def read_data():
    print("Reading data...")
    df = pd.DataFrame(columns = ['Path', 'Label'])
    train_labels = get_train_labels()
    i=0
    for j in range(len(train_labels[0].values)):
        try:
            for img in os.listdir(train_labels[0][j]):
                if '_' not in list(img):
                    df.loc[i] = [train_labels[0][j]+img, int(train_labels[1][j])]
                    i=i+1
                    # print(img)
        except:
            print("Error finding: "+str(train_labels[0][j]))
            pass

    # To convert labels into
    df['Label'] = pd.to_numeric(df['Label'])
    print("Done!")
    print("Done! Total number of images = "+str(df.shape[0]))
    return df

def get_train_test_data():
    print("Generating Validation samples...")
    data = read_data()
    train_data,val_data,_,_=train_test_split(data, data['Label'] , test_size=0.1, random_state=42)
    print("Done!")
    print("Number of Training images = "+str(train_data.shape[0]))
    print("Number of validation images = "+str(val_data.shape[0]))
    return train_data, val_data


def get_datagen():
    print("Creating generators...")
    train_data, val_data = get_train_test_data();

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.2,
        rotation_range=5,
        horizontal_flip=True)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe = train_data,
        directory  = '.',
        x_col = "Path",
        y_col = "Label",
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='other',
        batch_size=32)

    val_datagen = ImageDataGenerator(
        rescale = 1. /255)
    val_generator = val_datagen.flow_from_dataframe(
        dataframe = val_data,
        directory  = '.',
        x_col = "Path",
        y_col = "Label",
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='other',
        batch_size=32)
    print("Done!")
    return train_generator, val_generator
