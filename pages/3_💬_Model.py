import pandas as pd
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sn
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

st.set_page_config(
    page_title="Model Logistic Regression",
    page_icon="💬",
)

# Judul
st.markdown(
    """
    <div style='text-align: center; font-size: 60px; font-weight: bold;'>
        Model Logistioc Regression
    </div>
    """,
    unsafe_allow_html=True
)

# Menampilkan logo Suzuki di sidebar
try:
    logo = Image.open("suzuki.png")
    with st.sidebar:
        st.image(logo, use_container_width=True)
except FileNotFoundError:
    with st.sidebar:
        st.write("Logo tidak ditemukan")


# Mengubah warna sidebar
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #e4f6ff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load data
data = pd.read_excel('data_clean.xlsx')
data["clean_text"] = data["comment"].str.lower(
).str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)


# Split data
X_raw = data["clean_text"]
y_raw = data["Labeling"]
X_train, X_test, y_train, y_test = train_test_split(
    X_raw.values, y_raw.values, test_size=0.2, random_state=42, stratify=y_raw)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
vectorizer.fit(X_train)

X_train_TFIDF = vectorizer.transform(X_train).toarray()
X_test_TFIDF = vectorizer.transform(X_test).toarray()
kolom = vectorizer.get_feature_names_out()
train_tf_idf = pd.DataFrame(X_train_TFIDF, columns=kolom)
test_tf_idf = pd.DataFrame(X_test_TFIDF, columns=kolom)

# Streamlit UI
option = st.selectbox("Pilih Model Logisic Regression :",
                      ('1000 Features', 'Best Features'))

# Feature Selection
if option == '1000 Features':
    chi2_features = SelectKBest(chi2, k=1000)
elif option == 'Best Features':
    chi2_features = SelectKBest(chi2, k=2199)

X_train_chi2 = chi2_features.fit_transform(train_tf_idf, y_train)
X_test_chi2 = chi2_features.transform(test_tf_idf)

st.write('Banyaknya fitur awal:', train_tf_idf.shape[1])
st.write('Banyaknya fitur setelah seleksi:', X_train_chi2.shape[1])

# Pilih model
if option == '1000 Features':
    model_path = 'model_lr_1000.pkl'
elif option == 'Best Features':
    model_path = 'model_lr_best.pkl'
else:
    st.error("Pilihan tidak valid. Pilih '1000' atau 'best'.")

# Load model
tl = pickle.load(open(model_path, 'rb'))

# Prediksi
y_pred = tl.predict(X_test_chi2)

# Hasil Prediksi
st.write("Classification Report:")
report = classification_report(
    y_test, y_pred, target_names=['Negatif', 'Netral', 'Positif'])
st.markdown(f"```\n{report}\n```")

# Confusion Matrix
st.markdown("## Confusion Matrix")
kolom = ['negatif', 'netral', 'positif']
confm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(confm, index=kolom, columns=kolom)
fig, ax = plt.subplots(figsize=(5, 4))
sn.heatmap(df_cm, cmap='Greens', annot=True, fmt=".0f", ax=ax)
ax.set_title('Confusion Matrix')
ax.set_xlabel('Prediksi')
ax.set_ylabel('Aktual')
st.pyplot(fig)


# Form input untuk prediksi
st.write("### Prediksi Sentimen")
input_text = st.text_area("Masukkan teks untuk diprediksi:")

if st.button("Prediksi"):
    if input_text:
        input_vector = vectorizer.transform([input_text]).toarray()
        input_chi2 = chi2_features.transform(input_vector)
        hasil_prediksi = tl.predict(input_chi2)[0]

        warna = "gray"
        if hasil_prediksi == 'positif':
            warna = "#4CAF50"  # Hijau
        elif hasil_prediksi == 'netral':
            warna = "#FFC107"  # Kuning
        elif hasil_prediksi == 'negatif':
            warna = "#F44336"  # Merah

        st.markdown(f"""<div style="background-color:{warna};padding:10px;border-radius:5px;text-align:center;color:white;font-size:20px;">
        Hasil Prediksi: {hasil_prediksi}
        </div>""", unsafe_allow_html=True)
    else:
        st.error("Masukkan teks terlebih dahulu!")
