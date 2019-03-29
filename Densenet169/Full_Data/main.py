import config
from preprocess import *
from densenet169 import *

import pandas as pd
import numpy as np
import os
from keras.models import Model
from keras.layers import Dense, Activation
from keras.callbacks import EarlyStopping, TensorBoard


# Load Data for training
train_generator, val_generator = get_datagen()

# Start Training
model = build_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
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

save_model(model)
