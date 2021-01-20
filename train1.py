import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorboard.plugins.hparams import api as hp


dataset_dir_base = '../data/MergedDataset/'
dataset_paths = [dataset_dir_base + split for split in ["train","val","test"]]
model_checkpoint_filepath = "Models/checkpoint_{}.h5"
log_dir_base = "logs/hparam_tuning/"  # "logs/fit/"

img_height = 512
img_width = img_height//2
batch_size = 40
epochs = 16

HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32, 64]))
HP_LEARNING_RATE = hp.HParam('lrate', hp.Discrete([1e-1, 1e-2, 1e-3]))
HP_LABEL_SMOOTHING = hp.HParam('label_smoothing', hp.Discrete([0.0, 0.1]))
#HP_REGULARIZER = hp.HParam('regularizer', hp.Discrete([tf.keras.regularizers.l1(0.01), tf.keras.regularizers.l2(0.01)]))
HP_REGULARIZER = hp.HParam('regularizer', hp.Discrete(['l1', 'l2']))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.0, 0.4, 0.8]))



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


def build_model(hparams):
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
    model.add(Dense(256, activation='relu', kernel_regularizer=hparams[HP_REGULARIZER]))
    model.add(Dropout(hparams[HP_DROPOUT]))
    model.add(Dense(32, activation='relu', kernel_regularizer=hparams[HP_REGULARIZER]))
    model.add(Dense(6, activation='softmax'))

    model.compile(
        loss=CategoricalCrossentropy(label_smoothing=hparams[HP_LABEL_SMOOTHING]),
        optimizer=Adam(learning_rate=hparams[HP_LEARNING_RATE]),
        metrics=['accuracy'])
    model.summary()
    return model


def train_model(hparams, log_id):
    model = build_model(hparams)
    generators = get_generators()
    num_samples = get_num_samples()
    log_dir = log_dir_base + log_id

    TB = TensorBoard(log_dir=log_dir, histogram_freq=1)
    HP = hp.KerasCallback(log_dir, hparams)
    ES = EarlyStopping(monitor='val_loss', patience=2)
    MC = ModelCheckpoint(filepath=model_checkpoint_filepath.format(log_id), verbose=1, save_best_only=True)

    history = model.fit_generator(
        generators[0],
        steps_per_epoch=np.ceil(num_samples[0] / batch_size),
        epochs=epochs,
        validation_data=generators[1],
        validation_steps=np.ceil(num_samples[1] / batch_size),
        callbacks=[TB, HP, MC]
    )

    return test_model(model)


def test_saved_model():
    model = tf.keras.models.load_model(model_checkpoint_filepath)
    test_model(model)


def test_model(trained_model):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        dataset_paths[2],
        target_size=(img_height, img_width),
		color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical')

    loss, score = trained_model.evaluate_generator(
        test_generator,
        steps=test_generator.n // batch_size + 1,
        verbose=2,
        callbacks=[])

    return score


def run(run_name, hparams):
    with tf.summary.create_file_writer(log_dir_base + run_name).as_default():
        hp.hparams(hparams)
        batch_size = hparams[HP_BATCH_SIZE]
        accuracy = train_model(hparams, run_name)
        tf.summary.scalar('accuracy', accuracy, step=1)


def main():
    session_num = 0
    for hp_batch_size in HP_BATCH_SIZE.domain.values:
        for lrate in HP_LEARNING_RATE.domain.values:
            for smoothing in HP_LABEL_SMOOTHING.domain.values:
                for regularizer in HP_REGULARIZER.domain.values:
                    for dropout in HP_DROPOUT.domain.values:
                        hparams = {
                            HP_BATCH_SIZE: hp_batch_size,
                            HP_LEARNING_RATE: lrate,
                            HP_LABEL_SMOOTHING: smoothing,
                            HP_REGULARIZER: regularizer,
                            HP_DROPOUT: dropout
                        }
                        log_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                        log_id = "run-{}-{}".format(log_id, session_num)
                        print('--- Starting trial: {}'.format(log_id ))
                        print({h.name: hparams[h] for h in hparams})
                        run(log_id, hparams)
                        session_num += 1


if __name__ == '__main__':
    main()
	