# Model CNN + Hyperparameter Tuning Super Cepat
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras_tuner as kt
import numpy as np

# -------------------------
# Preprocessing CIFAR-10 (subset untuk cepat)
# -------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Ambil subset 10.000 data untuk cepat
subset_size = 10000
x_train = x_train[:subset_size]
y_train = y_train[:subset_size]

# Resize & normalisasi
x_train = tf.image.resize(x_train, (64, 64)) / 255.0
x_test = tf.image.resize(x_test, (64, 64)) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------------
# Fungsi build model untuk Keras Tuner
# -------------------------
def build_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))
    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())

    # Tunable dense layer 1
    units1 = hp.Int('units1', min_value=32, max_value=64, step=16)  # lebih kecil
    model.add(layers.Dense(units1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Tunable dense layer 2
    units2 = hp.Int('units2', min_value=32, max_value=64, step=16)
    model.add(layers.Dense(units2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Tunable dropout
    dropout_rate = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
    model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(10, activation='softmax'))

    # Tunable learning rate
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------
# Jalankan Hyperparameter Tuning Cepat
# -------------------------
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=2,               # lebih kecil
    executions_per_trial=1,
    directory='tuning_dir',
    project_name='cifar10_tuning_super_fast'
)

print("Mulai hyperparameter tuning super cepat...")
tuner.search(
    x_train, y_train,
    epochs=2,                    # lebih cepat
    validation_data=(x_test, y_test),
    batch_size=128                # batch lebih besar
)

# -------------------------
# Ambil dan Evaluasi Model Terbaik
# -------------------------
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

loss, acc = best_model.evaluate(x_test, y_test)
print(f"Akurasi terbaik: {acc:.4f}")

# -------------------------
# Simpan model terbaik
# -------------------------
best_model_filename = "best_model_vgg16_tuned_super_fast.keras"
best_model.save(best_model_filename)
print(f"Model terbaik tersimpan sebagai: {best_model_filename}")
