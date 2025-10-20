# Model CNN + Hyperparameter Tuning
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras_tuner as kt

# -------------------------
# Preprocessing CIFAR-10
# -------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = tf.image.resize(x_train, (64, 64)) / 255.0
x_test = tf.image.resize(x_test, (64, 64)) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# -------------------------
# Fungsi untuk build model (Keras Tuner)
# -------------------------
def build_model(hp):
    """
    Membuat model berbasis VGG16 dengan hyperparameter yang bisa dituning.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))
    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())

    # Tunable dense layer 1
    units1 = hp.Int('units1', min_value=64, max_value=256, step=32)
    model.add(layers.Dense(units1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Tunable dense layer 2
    units2 = hp.Int('units2', min_value=64, max_value=256, step=32)
    model.add(layers.Dense(units2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Tunable dropout
    dropout_rate = hp.Float('dropout', min_value=0.2, max_value=0.6, step=0.1)
    model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(10, activation='softmax'))

    # Tunable learning rate
    lr = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# -------------------------
# Jalankan Hyperparameter Tuning
# -------------------------
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,               # jumlah kombinasi hyperparameter yang dicoba
    executions_per_trial=1,
    directory='tuning_dir',
    project_name='cifar10_tuning'
)

print("Mulai hyperparameter tuning...")
tuner.search(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

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
best_model_filename = "best_model_vgg16_tuned.keras"
best_model.save(best_model_filename)
print(f"Model terbaik tersimpan sebagai: {best_model_filename}")
