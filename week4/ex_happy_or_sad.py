import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab happy-or-sad.zip from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = "../datasets/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("../tmp/h-or-s")
zip_ref.close()

# GRADED FUNCTION: train_happy_sad_model


def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(  # your code):
            # Your Code
            tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') > DESIRED_ACCURACY):
                print("\nReached %2.1f%% accuracy, so stopping training!!" %
                      (DESIRED_ACCURACY*100))
                self.model.stop_training = True

    callbacks = myCallback()

    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
        # Your Code Here
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                               input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(  # Your Code Here #
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=0.001),
        metrics=['accuracy'])

    # This code block should create an instance of an ImageDataGenerator called train_datagen
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)  # Your Code Here

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        # Your Code Here
        '../tmp/h-or-s/',
        target_size=(150, 150),
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary'
    )
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(
        # Your Code Here
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        callbacks=[callbacks],
        verbose=1
    )
    # model fitting
    return history.history['accuracy'][-1]


train_happy_sad_model()
