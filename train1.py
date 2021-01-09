import json
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataset_dir_base = '../data/MergedDataset/'
dataset_paths = [dataset_dir_base + split for split in ["train","val","test"]]
model_checkpoint_filepath = "checkpoints.h5"
model_history_filepath = "history.json.txt"
log_dir_base = "logs/fit/"

img_height = 512
img_width = img_height//2
batch_size = 40
epochs = 16
learn_rate = 0.001


def get_generators():
    test_gen = ImageDataGenerator(rescale=1. / 255)
    train_gen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,        # optional augmentation
        zoom_range=0.2,         # optional augmentation
        rotation_range=0.2,     # optional augmentation
        horizontal_flip=True,   # optional augmentation
    )

    train_generator = train_gen.flow_from_directory(
        dataset_paths[0],
        target_size=(img_height, img_width),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    validation_generator = test_gen.flow_from_directory(
        dataset_paths[1],
        target_size=(img_height, img_width),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    return [train_generator, validation_generator]


def get_num_samples():
    def get_num_of_all_files(path):
        return sum([len(files) for r, d, files in os.walk(path)])
    return [get_num_of_all_files(split) for split in dataset_paths]


def build_model():
    conv_base = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=(img_height, img_width, 3),
        pooling='avg'
    )
    for layer in conv_base.layers[:-6]:
        layer.trainable = False

    model = tf.keras.Sequential()
    model.add(conv_base)
    model.add(Dense(256, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=learn_rate),
        metrics=['accuracy'])
    model.summary()
    return model


def train_model(save_history=True):
    model = build_model()
    generators = get_generators()
    num_samples = get_num_samples()
    log_dir = log_dir_base + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
	
    history = model.fit_generator(
        generators[0],
        steps_per_epoch=np.ceil(num_samples[0] / batch_size),
        epochs=epochs,
        validation_data=generators[1],
        validation_steps=np.ceil(num_samples[1] / batch_size),
        #verbose = 2
        callbacks=[tensorboard_callback,
                   ModelCheckpoint(filepath=model_checkpoint_filepath, verbose=1, save_best_only=True)]
    )

    if save_history:
        save_history_to_file(history)


def save_history_to_file(history):
    json.dump(history.history, open(model_history_filepath, 'w'))


def test_saved_model():
    model = tf.keras.models.load_model(model_checkpoint_filepath)
    test_model(model)


def test_model(trained_model):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        dataset_paths[2],
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    score = trained_model.evaluate_generator(
        test_generator,
        steps=test_generator.n // batch_size + 1,
        verbose=2,
        callbacks=[])

    print(score)
    print(trained_model.metrics_names)


if __name__ == '__main__':
    train_model()
    #test_saved_model()