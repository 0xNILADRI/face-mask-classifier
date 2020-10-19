import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# configuration
batch_size = 128
img_height = 250
img_width = 250

## loading training data
training_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'data/',
    validation_split=0.2,
    seed = 42,
    subset= "training",
    image_size= (img_height, img_width),
    batch_size=batch_size

)

## loading testing data
testing_ds = tf.keras.preprocessing.image_dataset_from_directory(
'data/',
    validation_split=0.2,
    seed = 42,
    subset= "validation",
    image_size= (img_height, img_width),
    batch_size=batch_size

)

class_names = training_ds.class_names

## Configuring dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
training_ds = training_ds.cache().prefetch(buffer_size=AUTOTUNE)
testing_ds = testing_ds.cache().prefetch(buffer_size=AUTOTUNE)

# we will try to save the model which has the highest validation accuracy overall
# save best model using vall accuracy
model_path = 'assets/model/model.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# building our own neural network

# cnn model
## lets define our CNN
model = tf.keras.models.Sequential([
                                    layers.experimental.preprocessing.Rescaling(1./255),
                                    layers.Conv2D(32, 3, activation='relu'),
                                    layers.MaxPooling2D(),
                                    layers.Conv2D(64, 3, activation='relu'),
                                    layers.MaxPooling2D(),
                                    layers.Conv2D(64, 3, activation='relu'),
                                    layers.MaxPooling2D(),
                                    layers.Conv2D(128, 3, activation='relu'),
                                    layers.MaxPooling2D(),
                                    layers.Conv2D(256, 3, activation='relu'),
                                    layers.MaxPooling2D(),
                                    layers.Flatten(),
                                    layers.Dense(128, activation='relu'),
                                    layers.Dropout(0.5),
                                    layers.Dense(128, activation='relu'),
                                    layers.Dropout(0.3),
                                    layers.Dense(2, activation= 'softmax')
])

# compile model
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

# train model
history = model.fit(training_ds,
                    validation_data = testing_ds,
                    epochs=15, 
                    verbose=1,
                    callbacks=callbacks_list)

# plot the accuracy
plt.figure(figsize=(12,8));
plt.plot(history.history['acc'], label='Training Accuracy');
plt.plot(history.history['val_acc'], label='Validation Accuracy');
plt.legend();
plt.title('Accuracy');
plt.show();
plt.savefig('assets/image/accVal');

# plot the loss
plt.figure(figsize=(12,8));
plt.plot(history.history['loss'], label='Training Loss');
plt.plot(history.history['val_loss'], label='Validation Loss');
plt.legend();
plt.title('Loss');
plt.show();
plt.savefig('assets/image/lossVal');