import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import bentoml


# Data loading and pre-processing
train_dir = './intel-image-classification/seg_train/seg_train'
val_dir = './intel-image-classification/seg_val'

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest')

train_ds = train_gen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='sparse'
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_ds = val_gen.flow_from_directory(
    val_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='sparse'
)

# Define Model
def create_model(
    input_shape = (150,150,3),
    base_model_name = 'Xception',
    trainable = False,
    learning_rate = 0.001,
    size_inner = 256,
    droprate=0.3,
    num_classes = len(train_ds.class_indices),
):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = trainable

    #model structure
    inputs = Input(shape=input_shape) #input layer
    base = base_model(inputs, training=False) #pre-training extract features
    vectors= GlobalAveragePooling2D()(base) #pooling, 4D to 2D
    dropout_vectors = Dropout(droprate)(vectors) #dropout: random abandon some features
    inner = Dense(size_inner, activation = 'relu')(dropout_vectors) #relu activation
    dropout_inner = Dropout(droprate)(inner) #dropout: enhanced normalization
    outputs = Dense(num_classes, activation='softmax')(dropout_inner)

    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    return model

# Set Callback
callbacks = [
    ModelCheckpoint("best_model_tf17.h5", save_best_only=True, monitor="val_loss", mode="min"),
    EarlyStopping(patience=5, monitor="val_loss", mode="min", restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5, monitor="val_loss", mode="min")
]

# Train Model
model = create_model()
history = model.fit(train_ds, epochs=50, validation_data=val_ds, callbacks=callbacks)

print("Model saved as best_model_tf17.h5")

# Save Keras model to BentoML repository
# bentoml.tensorflow.save_model("image_classifier_model", model)
# print("Model has daved into BentoML！")