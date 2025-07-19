import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from io import BytesIO
import nltk
import requests
import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --- NLTK Data Downloads --- 
@st.cache_resource
def download_nltk_data():
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('sentiwordnet')
    nltk.download('vader_lexicon')

download_nltk_data()

# --- Streamlit App Starts Here ---
st.title("🤖 Sentiment Analysis of Bitcoin Comments Dashboard")
st.sidebar.title("Navigation & Filters")
st.markdown("""
    ### Interactive Bitcoin Comment Analysis Dashboard
    Jelajahi pola sentimen dari komentar terkait Bitcoin dengan visualisasi interaktif.
    """)
st.sidebar.markdown("Filter data dan konfigurasikan visualisasi:")

# --- Definisi Nama Kolom ---
komentar_col = 'komentar'  # Kolom komentar asli
cleaned_col_data = 'cleaned_text'  # Kolom komentar yang sudah diproses
sentiment_col_textblob = 'sentiment_textblob'  # Kolom sentiment untuk TextBlob
sentiment_col_vader = 'sentiment_vader'  # Kolom sentiment untuk Vader
sentiment_col_sentiwordnet = 'sentiment_sentiwordnet'  # Kolom sentiment untuk SentiWordNet

# --- Fungsi untuk Memuat Data ---
@st.cache_data(persist=True)
def load_data(dataset_type='sentiment_analysis'):
    """
    Function to load the dataset based on the selected sentiment analysis method.
    """
    dataset_map = {
        'texblob': "sentiment_textblob.csv",
        'vader': "sentiment_vader.csv",
        'sentiwordnet': "sentiment_sentiwordnet.csv"
    }
    try:
        data = pd.read_csv(dataset_map[dataset_type])
    except (FileNotFoundError, KeyError):
        st.error(f"ERROR: File for {dataset_type} not found.")
        st.stop()
    return data

# --- Struktur Tab Dashboard ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Text Analysis", "Sentiment Prediction", "ML Sentiment Prediction", "DL Sentiment Prediction"
])

# ========== Tab 1: Overview ==========
with tab1:
    st.header("Statistik Umum")
    
    # --- Pilihan untuk distribusi sentimen ---
    st.subheader("Distribusi Sentimen Bitcoin Comments")
    
    # --- Sentiment Analysis Method Selection in Main Content ---
    method = st.selectbox("Pilih Metode Analisis Sentimen", ['TexBlob', 'Vader', 'SentiWordNet'])

    # Load dataset based on the sentiment method selected
    data = load_data(dataset_type=method.lower())

    # Pilih kolom sentiment sesuai metode yang dipilih
    if method == 'TexBlob':
        sentiment_column = sentiment_col_textblob
    elif method == 'Vader':
        sentiment_column = sentiment_col_vader
    else:  # SentiWordNet
        sentiment_column = sentiment_col_sentiwordnet
    
    # Group by sentiment and count
    sentiment_count = data[sentiment_column].value_counts().reset_index()
    sentiment_count.columns = ['Sentiment', 'Count']

    # Pilihan untuk visualisasi
    chart_type = st.selectbox("Pilih jenis visualisasi", ['Bar chart', 'Pie chart'])

    if chart_type == 'Bar chart':
        # Plotly bar chart for sentiment distribution with hover information
        fig = px.bar(sentiment_count, 
                     x='Sentiment', 
                     y='Count', 
                     color='Sentiment', 
                     hover_data={'Sentiment': True, 'Count': True}, 
                     labels={'Sentiment': 'Sentiment Type', 'Count': 'Number of Comments'}, 
                     title=f"Distribusi Sentimen {method} Bitcoin Comments",
                     height=400)

        fig.update_layout(xaxis_title='Sentiment', yaxis_title='Number of Comments')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == 'Pie chart':
        # Plotly pie chart for sentiment distribution with hover information
        fig = px.pie(sentiment_count, 
                     values='Count', 
                     names='Sentiment', 
                     hover_data={'Sentiment': True, 'Count': True}, 
                     title=f"Distribusi Sentimen {method} Bitcoin Comments")

        st.plotly_chart(fig, use_container_width=True)

    # --- Tabel Akurasi Model Machine Learning dari CSV ---
    st.subheader("Hasil Akurasi Model Machine Learning")
    st.info("Tabel berikut menampilkan akurasi berbagai model machine learning dengan 3 metode feature extraction (BoW, TF-IDF, Word2Vec).")

    try:
        accuracy_df = pd.read_csv("machinelearning_accuracy_results.csv")
        st.dataframe(accuracy_df, use_container_width=True)

        accuracy_long = accuracy_df.melt(
            id_vars=['Model'],
            value_vars=['Accuracy_BoW', 'Accuracy_TFIDF', 'Accuracy_Word2Vec'],
            var_name='Feature Extraction',
            value_name='Accuracy'
        )
        accuracy_long['Feature Extraction'] = accuracy_long['Feature Extraction'].replace({
            'Accuracy_BoW': 'BoW',
            'Accuracy_TFIDF': 'TF-IDF',
            'Accuracy_Word2Vec': 'Word2Vec'
        })

        fig_acc = px.bar(
            accuracy_long,
            x='Model',
            y='Accuracy',
            color='Feature Extraction',
            barmode='group',
            title="Akurasi Model Machine Learning per Feature Extraction",
            text_auto='.2f',
            height=400
        )
        fig_acc.update_layout(yaxis_title='Akurasi', xaxis_title='Model')
        st.plotly_chart(fig_acc, use_container_width=True)

    except Exception as e:
        st.warning(f"Gagal memuat data akurasi model: {e}")

    # --- Tabel Akurasi Model Deep Learning dari CSV ---
    st.subheader("Hasil Akurasi Model Deep Learning")
    st.info("Tabel berikut menampilkan akurasi berbagai arsitektur deep learning (misal: LSTM, GRU, CNN, dsb).")

    try:
        dl_acc_df = pd.read_csv("deep_learning_model_performance.csv")
        st.dataframe(dl_acc_df, use_container_width=True)

        # Asumsi kolom: Model, Accuracy, Precision, Recall, F1 (atau sesuaikan dengan file Anda)
        # Jika hanya Model dan Accuracy:
        if 'Accuracy' in dl_acc_df.columns:
            fig_dl_acc = px.bar(
                dl_acc_df,
                x='Model',
                y='Accuracy',
                color='Model',
                text_auto='.2f',
                title="Akurasi Model Deep Learning",
                height=400
            )
            fig_dl_acc.update_layout(yaxis_title='Akurasi', xaxis_title='Model', showlegend=False)
            st.plotly_chart(fig_dl_acc, use_container_width=True)
        # Jika ada beberapa metrik (Accuracy, Precision, Recall, F1)
        else:
            dl_acc_long = dl_acc_df.melt(
                id_vars=['Model'],
                var_name='Metric',
                value_name='Score'
            )
            fig_dl_acc = px.bar(
                dl_acc_long,
                x='Model',
                y='Score',
                color='Metric',
                barmode='group',
                text_auto='.2f',
                title="Performa Model Deep Learning",
                height=400
            )
            fig_dl_acc.update_layout(yaxis_title='Score', xaxis_title='Model')
            st.plotly_chart(fig_dl_acc, use_container_width=True)

    except Exception as e:
        st.warning(f"Gagal memuat data akurasi deep learning: {e}")

# ========== Tab 2: Text Analysis ==========
with tab2:
    st.header("Analisis Tekstual")
    
    st.subheader("Word Cloud (Gambar)")

    st.info("Pilih metode analisis sentimen untuk melihat word cloud per sentimen.")

    # Pilihan metode wordcloud
    wc_method = st.selectbox(
        "Pilih Metode Word Cloud",
        ['TextBlob', 'Vader', 'SentiWordNet'],
        key="wc_method"
    )

    # Mapping file gambar untuk setiap metode dan sentimen
    wordcloud_images = {
        'TextBlob': {
            'positive': 'img/textblob/textblob_positive.png',
            'neutral': 'img/textblob/textblob_neutral.png',
            'negative': 'img/textblob/textblob_negative.png'
        },
        'Vader': {
            'positive': 'img/vader/vader_positive.png',
            'neutral': 'img/vader/vader_neutral.png',
            'negative': 'img/vader/vader_negative.png'
        },
        'SentiWordNet': {
            'positive': 'img/sentiwordnet/sentiwordnet_positive.png',
            'neutral': 'img/sentiwordnet/sentiwordnet_neutral.png',
            'negative': 'img/sentiwordnet/sentiwordnet_negative.png'
        }
    }

    for label in ['positive', 'neutral', 'negative']:
        img_path = wordcloud_images[wc_method][label]
        st.subheader(f"Word Cloud untuk Sentimen {label.capitalize()} ({wc_method})")
        if os.path.exists(img_path):
            st.image(img_path, caption=f"Word Cloud Sentimen {label.capitalize()} - {wc_method}", use_container_width=True)
            with open(img_path, "rb") as img_file:
                st.download_button(
                    label=f"Unduh Word Cloud {label.capitalize()} ({wc_method})",
                    data=img_file,
                    file_name=img_path,
                    mime="image/png",
                    key=f"download_wc_{wc_method}_{label}"
                )
        else:
            st.warning(f"Gambar word cloud untuk sentimen {label.capitalize()} ({wc_method}) tidak ditemukan.")

# ========== Tab 3: Sentiment Prediction ==========
with tab3:
    st.header("Prediksi Sentimen Komentar")
    st.info("Masukkan kata atau kalimat untuk diprediksi sentimennya. Pilih metode analisis sentimen yang diinginkan.")

    pred_method = st.selectbox(
        "Pilih Metode Prediksi Sentimen",
        ['Vader', 'TextBlob'],
        key="pred_method"
    )

    user_input = st.text_area("Masukkan kata atau kalimat di sini", "")

    if st.button("Prediksi Sentimen"):
        if user_input.strip() == "":
            st.warning("Silakan masukkan kata atau kalimat terlebih dahulu.")
        else:
            sentiment = None
            score = None

            if pred_method == "Vader":
                from nltk.sentiment import SentimentIntensityAnalyzer
                sia = SentimentIntensityAnalyzer()
                scores = sia.polarity_scores(user_input)
                compound = scores['compound']
                score = compound
                if compound >= 0.05:
                    sentiment = "Positive"
                    st.success(f"Sentimen: {sentiment} (score: {compound:.2f})")
                elif compound <= -0.05:
                    sentiment = "Negative"
                    st.error(f"Sentimen: {sentiment} (score: {compound:.2f})")
                else:
                    sentiment = "Neutral"
                    st.info(f"Sentimen: {sentiment} (score: {compound:.2f})")

            elif pred_method == "TextBlob":
                from textblob import TextBlob
                tb = TextBlob(user_input)
                polarity = tb.sentiment.polarity
                score = polarity
                if polarity > 0.05:
                    sentiment = "Positive"
                    st.success(f"Sentimen: {sentiment} (score: {polarity:.2f})")
                elif polarity < -0.05:
                    sentiment = "Negative"
                    st.error(f"Sentimen: {sentiment} (score: {polarity:.2f})")
                else:
                    sentiment = "Neutral"
                    st.info(f"Sentimen: {sentiment} (score: {polarity:.2f})")
# ========== Tab 4: ML Sentiment Prediction ==========
with tab4:
    st.header("Prediksi Sentimen (Machine Learning - Logistic Regression + BoW)")
    st.info("Masukkan kata atau kalimat untuk diprediksi sentimennya menggunakan model Logistic Regression (Bag of Words).")

    ml_input = st.text_area("Masukkan kata atau kalimat di sini untuk prediksi ML", key="ml_input")

    # Load model & vectorizer
    @st.cache_resource
    def load_ml_model():
        with open("best_lr_bow.pkl", "rb") as f:
            model_data = pickle.load(f)
        # Model file harus berupa dict: {'model': ..., 'vectorizer': ...}
        return model_data['model'], model_data['vectorizer']

    try:
        model, vectorizer = load_ml_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        model_loaded = False

    if st.button("Prediksi Sentimen ML") and model_loaded:
        if ml_input.strip() == "":
            st.warning("Silakan masukkan kata atau kalimat terlebih dahulu.")
        else:
            X_input = vectorizer.transform([ml_input])
            pred = model.predict(X_input)[0]
            # Mapping label angka ke string
            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            label = label_map.get(pred, str(pred))
            if label == "Positive":
                st.success(f"Sentimen: Positive")
            elif label == "Negative":
                st.error(f"Sentimen: Negative")
            else:
                st.info(f"Sentimen: Neutral")

# ========== Tab 5: DL Sentiment Prediction ==========
with tab5:
    st.header("Prediksi Sentimen (Deep Learning - LSTM)")
    st.info("Masukkan kata atau kalimat untuk diprediksi sentimennya menggunakan model LSTM (deep learning).")

    dl_input = st.text_area("Masukkan kata atau kalimat di sini untuk prediksi DL", key="dl_input")

    # Load model & tokenizer
    @st.cache_resource
    def load_dl_model():
        model = tf.keras.models.load_model("best_model_lstm_sentiment.h5")
        import joblib
        tokenizer = joblib.load("tokenizer.pkl")  # <-- pastikan file ini benar
        return model, tokenizer

    try:
        dl_model, dl_tokenizer = load_dl_model()
        dl_model_loaded = True
    except Exception as e:
        st.error(f"Gagal memuat model/tokenizer LSTM: {e}")
        dl_model_loaded = False

    maxlen = 100  # Ganti sesuai maxlen saat training

    if st.button("Prediksi Sentimen DL") and dl_model_loaded:
        if dl_input.strip() == "":
            st.warning("Silakan masukkan kata atau kalimat terlebih dahulu.")
        else:
            # Preprocessing input
            sequences = dl_tokenizer.texts_to_sequences([dl_input])
            padded = pad_sequences(sequences, maxlen=maxlen)
            pred = dl_model.predict(padded)
            # Asumsi output: 3 kelas (Negative, Neutral, Positive)
            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            pred_label = pred.argmax(axis=1)[0]
            label = label_map.get(pred_label, str(pred_label))
            score = float(pred[0][pred_label])
            if label == "Positive":
                st.success(f"Sentimen: Positive (confidence: {score:.2f})")
            elif label == "Negative":
                st.error(f"Sentimen: Negative (confidence: {score:.2f})")
            else:
                st.info(f"Sentimen: Neutral (confidence: {score:.2f})")

# --- Penambahan Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("Ekspor Data")
if st.sidebar.button("Unduh Komentar Berlabel sebagai CSV"):
    sentiwordnet_df = pd.read_csv("sentiment_sentiwordnet.csv")
    csv = sentiwordnet_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Unduh CSV",
        data=csv,
        file_name="Komentar Berlabel.csv",
        mime="text/csv"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Tentang Penulis")

image_url = "https://avatars.githubusercontent.com/u/98022263?v=4"
try:
    response = requests.get(image_url)
    response.raise_for_status()
    image = response.content
    st.sidebar.image(image, caption="Maulana Imanulhaq (Fancyyy21)")
except requests.exceptions.RequestException as e:
    st.sidebar.error(f"Gagal memuat gambar profil: {e}")

st.sidebar.info("""
    **Fitur Dashboard:**
    - Plot distribusi sentimen interaktif dengan pilihan metode analisis (TexBlob, Vader, SentiWordNet)
    - Kemampuan ekspor data
    - Word cloud untuk visualisasi kata-kata yang sering muncul per sentimen
    - Prediksi sentimen untuk input teks menggunakan berbagai metode
    - Tabel akurasi model machine learning dan deep learning
    - Visualisasi performa model machine learning dan deep learning
    - Pilihan untuk menampilkan word cloud per sentimen
    - Pilihan untuk memprediksi sentimen komentar menggunakan model machine learning (Logistic Regression) dan deep learning (LSTM)
    - Menampilkan contoh komentar asli teratas per sentimen
""")
