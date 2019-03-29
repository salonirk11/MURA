import config

from keras.preprocessing.image import ImageDataGenerator, load_img, image
from keras.applications.densenet import DenseNet169,preprocess_input
from keras.initializers import glorot_uniform
from keras.models import Model
from keras.layers import Dense, Activation
from keras import regularizers
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import model_from_json

def build_model():
    print("Building Model...")
    base_model = DenseNet169(
                          weights=WEIGHTS,
                          input_shape=(224, 224, CHANNELS),
                          pooling='max',
                          classes=1000)

    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)

    predictions = Dense(NUM_CLASSES,activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    print("Done!")
    return model

def save_model(model):
    print("Saving model...")
    with open(ARCH_PATH,'r') as f:
        model=model_from_json(f.read())
        model.load_weights(WEIGHTS_PATH)
    print("Done!")
