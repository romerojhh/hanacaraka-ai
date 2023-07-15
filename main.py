import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications.inception_v3 import InceptionV3
from kaggle.api.kaggle_api_extended import KaggleApi

data_dir = 'hanacaraka-new'
train_dir = 'hanacaraka-new/v3/v3/train'
val_dir = 'hanacaraka-new/v3/v3/val'
dataset_id = "phiard/aksara-jawa"
kaggle_api = KaggleApi()
kaggle_api.authenticate()
kaggle_api.dataset_download_files(dataset=dataset_id, path=data_dir, unzip=True)

image_height = 150
image_width = 150
batch_size = 32


def extract_file(path):
    data = image_dataset_from_directory(
        directory=path,
        label_mode='categorical',
        image_size=(image_height, image_width),
        batch_size=batch_size,
        color_mode='grayscale'
    )
    return data


def setup_model():
    image_augmentation = tf.keras.models.Sequential([
        tf.keras.layers.RandomRotation(factor=0.2, fill_mode='nearest', input_shape=(image_height, image_width, 1)),
        tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor=0.2, fill_mode='nearest'),
        tf.keras.layers.RandomZoom(height_factor=0.2, fill_mode='nearest'),
    ])

    model = tf.keras.models.Sequential([
        # normalize the data
        # tf.keras.layers.Rescaling(scale=1. / 255, input_shape=(image_height, image_width, 1)),
        image_augmentation,
        # 1st convolution
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        # The fourth convolution
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(20, activation='softmax')
    ])

    return model


def is_image_corrupted(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return False
    except (IOError, SyntaxError):
        return True


# Recursive function to iterate through directories
def check_images(directory):
    count = 0
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".png"):
                image_path = os.path.join(root, filename)
                if is_image_corrupted(image_path):
                    count += 1
                    print(f"The image '{image_path}' is corrupted.")

    print(f'{count} file(s) corrupted')


def pretrained():
    pre_trained_model = InceptionV3(
        input_shape=(image_height, image_width, 3),
        include_top=False
    )

    # Freezing the entire model to prevent it from retraining
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Get the output from 'mixed7' layer from inception_v3
    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    return last_layer.output, pre_trained_model


def plot(hist):
    history = hist.history
    history['epoch'] = hist.epoch

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.plot(history['epoch'], history['loss'], label='Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss', color='orange')
    plt.legend()

    plt.subplot(122)
    plt.plot(history['epoch'], history['accuracy'], label='Acc')
    plt.plot(history['epoch'], history['val_accuracy'], label='Val Acc', color='orange')
    plt.legend()

    return plt.show()


if __name__ == '__main__':
    # check_images(train_dir)
    # check_images(val_dir)
    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    validation_data = extract_file(val_dir)
    normalized_validation_data = validation_data.map(lambda x, y: (normalization_layer(x), y))

    train_data = extract_file(train_dir)
    normalized_train_data = train_data.map(lambda x, y: (normalization_layer(x), y))

    # last_output = pretrained()
    #
    # x = tf.keras.layers.Flatten()(last_output[0])
    # x = tf.keras.layers.Dense(256, activation='relu')(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    # x = tf.keras.layers.Dense(20, activation='softmax')(x)

    model = setup_model()
    # model = tf.keras.Model(last_output[1].input, x)
    model.summary()

    # initial_learning_rate = 0.0001
    # decay_steps = 1000
    # decay_rate = 0.1
    #
    # # learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    # #     initial_learning_rate, decay_steps, decay_rate)
    #
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        loss='categorical_crossentropy',
        metrics=['accuracy'],
        optimizer='adam'
    )

    hist = model.fit(
        normalized_train_data,
        epochs=50,
        validation_data=normalized_validation_data
    )
    plot(hist)
