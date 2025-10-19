from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras_tuner as kt

# -------------------------
# Load CIFAR-10
# -------------------------
def load_data(img_size=(64,64)):
    """
    Load CIFAR-10 dataset dan preprocessing.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = tf.image.resize(x_train, img_size) / 255.0
    x_test = tf.image.resize(x_test, img_size) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# -------------------------
# Model CNN biasa
# -------------------------
def create_model(input_shape=(64,64,3), num_classes=10):
    """
    Membuat model CNN berbasis VGG16 pretrained.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, epochs=5):
    """
    Melatih model dengan dataset CIFAR-10.
    """
    (x_train, y_train), (x_test, y_test) = load_data()
    history = model.fit(x_train, y_train, epochs=epochs,
                        validation_data=(x_test, y_test))
    return history

def save_model(model, filename):
    """
    Menyimpan model ke file.
    """
    model.save(filename)
    print(f"Model berhasil disimpan sebagai: {filename}")

def load_model(filename):
    """
    Memuat model dari file.
    """
    model = tf.keras.models.load_model(filename)
    print(f"Model berhasil dimuat dari: {filename}")
    return model

# -------------------------
# Hyperparameter Tuning
# -------------------------
def build_model(hp):
    """
    Fungsi build model untuk Keras Tuner.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))
    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())

    # Hyperparameter: jumlah neuron Dense layer pertama
    units1 = hp.Int('units1', min_value=64, max_value=256, step=32)
    model.add(layers.Dense(units1))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Hyperparameter: jumlah neuron Dense layer kedua
    units2 = hp.Int('units2', min_value=64, max_value=256, step=32)
    model.add(layers.Dense(units2))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    # Hyperparameter: Dropout rate
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.6, step=0.1)
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(10, activation='softmax'))

    # Hyperparameter: learning rate
    lr = hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def tune_model(max_trials=10, executions_per_trial=1, epochs=5):
    """
    Melakukan hyperparameter tuning menggunakan Keras Tuner dan mengembalikan model terbaik.
    """
    (x_train, y_train), (x_test, y_test) = load_data()

    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='tuning_dir',
        project_name='cifar10_tuning'
    )

    tuner.search(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.summary()

    loss, acc = best_model.evaluate(x_test, y_test)
    print(f"Akurasi terbaik setelah tuning: {acc:.4f}")

    best_model.save("best_model_vgg16_tuned.keras")
    print("Model terbaik disimpan sebagai best_model_vgg16_tuned.keras")

    return best_model
    