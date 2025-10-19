from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Preprocessing CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = tf.image.resize(x_train, (64,64)) / 255.0
x_test = tf.image.resize(x_test, (64,64)) / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Menggunakan model dasar VGG16 tanpa top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))
base_model.trainable = False # Membekukan layer pretrained

model_combined = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Kompilasi Model
model_combined.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# Melatih model dan menyimpan riwayat proses pelatihan (akurasi dan loss)
history_combined = model_combined.fit(x_train, y_train, epochs=5,validation_data=(x_test, y_test))

# Save & Load Model
model_filename = 'cifar10_cnn_model_without_hyperparameter_tuning.keras'
model_combined.save(model_filename)
# Save entire model
print(f"Model berhasil disimpan sebagai: {model_filename}")
# Memuat kembali model
loaded_model = tf.keras.models.load_model(model_filename)
print("\nModel berhasil dimuat kembali.")
# Tampilkan ringkasan model
loaded_model.summary()
