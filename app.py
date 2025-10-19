import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
from model import create_model, train_model, save_model, load_model, tune_model

# Kelas CIFAR-10
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Pastikan folder model ada
MODEL_DIR = './model'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

import streamlit as st

tab1, tab2 = st.tabs(["Main", "Group Information"])

with tab1:
    st.header("CIFAR-10 Image Prediction using Convolutional Neural Network (CNN)")
    # Upload gambar untuk prediksi
    st.text('Pastikan image yang diupload termasuk dari 10 kelas CIFAR-10')
    st.text('10 kelas tersebut yakni: airplane, automobile, bird, cat, deer, dog, frog, horse, ship dan truck')
    st.text('Jika anda memasukkan gambar selain 10 kelas tersebut, model tetap akan memaksa memprediksi salah satu dari 10 kelas itu. Tapi hasil prediksi bisa tidak valid.')

    st.link_button('Klik ini untuk melihat referensi mengenai cifar-10', 'https://www.cs.toronto.edu/~kriz/cifar.html', help=None, type="secondary", icon=None, disabled=False, use_container_width=None, width="content")

    # Pilih mode utama
    mode = st.radio("Pilih Mode:", ["Gunakan Model yang Disimpan", "Latih Model Baru"])

    model = None

    # -------------------------------
    # Mode: Gunakan Model yang Disimpan
    # -------------------------------
    if mode == "Gunakan Model yang Disimpan":
        st.text("Gunakan model yang sudah tersimpan untuk prediksi gambar.")

        # Pilih file model yang sudah tersimpan
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.keras')]
        if model_files:
            selected_model_file = st.selectbox("Pilih file model", model_files)
            model_path = os.path.join(MODEL_DIR, selected_model_file)
            model = load_model(model_path)
        else:
            st.warning("Tidak ada file model ditemukan saat ini. Silakan latih model terlebih dahulu.")

        # Upload gambar untuk prediksi
        uploaded_file = st.file_uploader("Unggah gambar (64x64 RGB)", type=["png", "jpg", "jpeg"])
        if uploaded_file and model is not None:
            img = Image.open(uploaded_file).convert('RGB')
            st.image(img, caption="Gambar yang diunggah", use_column_width=True)

            # Preprocessing
            img = img.resize((64,64))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            # Prediksi
            pred = model.predict(img_array)
            pred_probs = pred[0]

            # Top 3 prediksi
            top3_indices = pred_probs.argsort()[-3:][::-1]
            st.subheader("Top 3 Prediksi:")
            for i, idx in enumerate(top3_indices):
                st.write(f"{i+1}. {cifar10_classes[idx]} - Confidence: {pred_probs[idx]:.2f}")

            # Peringatan jika confidence rendah
            threshold = 0.5
            if pred_probs[top3_indices[0]] < threshold:
                st.warning("⚠️ Model kurang yakin. Gambar mungkin bukan salah satu kelas CIFAR-10.")

            # Visualisasi distribusi probabilitas
            st.subheader("Distribusi Probabilitas 10 Kelas")
            df = pd.DataFrame({"Class": cifar10_classes, "Probability": pred_probs}).sort_values("Probability", ascending=True)
            fig, ax = plt.subplots()
            ax.barh(df["Class"], df["Probability"], color='skyblue')
            ax.set_xlabel("Probability")
            ax.set_title("Prediksi CIFAR-10")
            st.pyplot(fig)

    # -------------------------------
    # Mode: Latih Model Baru
    # -------------------------------
    elif mode == "Latih Model Baru":
        st.info("Pilih strategi training: model biasa atau hyperparameter tuning.")

        strategy = st.radio("Strategi Training:", ["Train Biasa", "Hyperparameter Tuning (Keras Tuner)"])

        # Input jumlah epoch
        epochs = st.number_input("Jumlah Epoch", min_value=1, max_value=50, value=5)

        # Input nama file model
        file_name = st.text_input("Nama file model (tanpa ekstensi .keras)")

        if st.button("Mulai Training"):
            if not file_name:
                st.warning("⚠️ Silakan masukkan nama file model.")
            else:
                model_path = os.path.join(MODEL_DIR, file_name + '.keras')
                if os.path.exists(model_path):
                    st.error(f"⚠️ File '{file_name}.keras' sudah ada. Silakan gunakan nama lain.")
                else:
                    if strategy == "Train Biasa":
                        st.info("Training model biasa dimulai...")
                        model = create_model()
                        with st.spinner("Training sedang berjalan..."):
                            history = train_model(model, epochs=epochs)
                            save_model(model, model_path)
                        st.success(f"Training selesai dan model disimpan sebagai: {model_path}")

                        # Tampilkan history
                        st.subheader("Training History")
                        st.line_chart(history.history['accuracy'], use_container_width=True)
                        st.line_chart(history.history['val_accuracy'], use_container_width=True)

                    elif strategy == "Hyperparameter Tuning (Keras Tuner)":
                        st.info("Hyperparameter tuning dimulai. Ini bisa memakan waktu lebih lama.")
                        with st.spinner("Tuning sedang berjalan..."):
                            best_model = tune_model(max_trials=10, executions_per_trial=1, epochs=epochs)
                            save_model(best_model, model_path)
                        st.success(f"Training selesai dan model terbaik disimpan sebagai: {model_path}")
with tab2:
    st.header("Group Information")
    st.text('Group : 6')
    st.text('Kelas : JPCA - LEC')
    st.text('Mata Kuliah : Artificial Intelligence')
    st.text('DAFFA ANAQI FARID - 2802514082')
    st.text('TAKESHI RIFQI PRAYOGA - 2802537016')
    st.text('WIDYA LESTARI SIMANJUNTAK - 2802555271')
    st.text('ADHIKA YUSUF ALIFIANSYAH - 2802549786')
    st.text('WAHYU NOVIANTI - 2802515476')
