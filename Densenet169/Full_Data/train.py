import config
import pandas as pd
import numpy as np
import os
import tqdm

from keras.preprocessing.image import ImageDataGenerator, load_img, image
from keras.applications.densenet import DenseNet169,preprocess_input
from keras.initializers import glorot_uniform
from keras.models import Model
from keras.layers import Dense, Activation
from keras import regularizers
from keras.callbacks import EarlyStopping, TensorBoard

from sklearn.model_selection import train_test_split


DATASET_DIR = 'MURA-v1.1/'
TRAIN_IMAGES = 'train_image_paths.csv'
TRAIN_LABELS = 'train_labeled_studies.csv'

NUM_CLASSES = 1
NUM_SAMPLES=32000
VAL_SIZE=3700
BATCH_SIZE=512


train_images = pd.read_csv(DATASET_DIR+TRAIN_IMAGES, header = None)
train_labels = pd.read_csv(DATASET_DIR+TRAIN_LABELS, header = None)



train_data = pd.DataFrame(columns = ['Path', 'Label'])
i=0
for j in range(len(train_labels[0].values)):
  try:
    for img in os.listdir(DATASET_DIR+train_labels[0][j]):
      if '_' not in list(img):
        train_data.loc[i] = [train_labels[0][j]+img, int(train_labels[1][j])]
        i=i+1
        print(img)
  except:
    print(DATASET_DIR+train_labels[0][j])
    pass



train_data['Label'] = pd.to_numeric(train_data['Label'])



train_data,val_data,_,_=train_test_split(train_data, train_data['Label'] , test_size=0.1, random_state=42)

def build_model():

    base_model = DenseNet169(
                          weights=None,
                          input_shape=(224, 224,3),
                          pooling='avg',
                          classes=1)

    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)

    predictions = Dense(1,activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    return model

model = build_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', 'mse'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    rotation_range=5,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_data,
    directory  = DATASET_DIR,
    x_col = "Path",
    y_col = "Label",
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='other',
    batch_size=32
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe = val_data,
    directory  = DATASET_DIR,
    x_col = "Path",
    y_col = "Label",
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='other',
    batch_size=32
)

model.fit_generator(
    generator = train_generator,
    steps_per_epoch = NUM_SAMPLES/BATCH_SIZE,
    epochs=10,
    verbose=1,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            min_delta=0.0001
        ),
        TensorBoard(
            log_dir='Graph_complete',
            histogram_freq=0,
            write_graph=True,
            write_images=True
        )
    ],
    validation_data=val_generator,
    validation_steps = VAL_SIZE/BATCH_SIZE,
    shuffle=True
)
