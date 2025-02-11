import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(
    page_title="Insight",
    page_icon="📚",
)

# Judul
st.markdown(
    """
    <div style='text-align: center; font-size: 60px; font-weight: bold;'>
        Insight Penelitian
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

st.sidebar.markdown(
    "<h1 style='text-align: center; color: black;'>Project nya Miwa San</h1>",
    unsafe_allow_html=True
)

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

# Memuat data
data = pd.read_excel('data_clean.xlsx')
data["clean_text"] = data["comment"].str.lower(
).str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)

# Memisahkan data latih dan data uji
X_raw = data["clean_text"]
y_raw = data["Labeling"]
X_train, X_test, y_train, y_test = train_test_split(
    X_raw.values, y_raw.values, test_size=0.2, random_state=42, stratify=y_raw)

# Mengubah data teks menjadi bilangan vektor
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
train_tf_idf = vectorizer.fit_transform(X_train)
test_tf_idf = vectorizer.transform(X_test)

# Visualisasi rata-rata nilai TF-IDF
train_mean_tfidf = pd.Series(train_tf_idf.mean(
    axis=0).A1, index=vectorizer.get_feature_names_out())
test_mean_tfidf = pd.Series(test_tf_idf.mean(
    axis=0).A1, index=vectorizer.get_feature_names_out())

fig, ax = plt.subplots()
train_mean_tfidf.nlargest(15).plot(kind='bar', ax=ax, color='blue')
ax.set_ylabel('Rata-rata Nilai TF-IDF')
ax.set_title('Top 15 TF-IDF pada Data Latih')
st.pyplot(fig)

fig, ax = plt.subplots()
test_mean_tfidf.nlargest(15).plot(kind='bar', ax=ax, color='red')
ax.set_ylabel('Rata-rata Nilai TF-IDF')
ax.set_title('Top 15 TF-IDF pada Data Uji')
st.pyplot(fig)

# Pivot tabel untuk distribusi sentimen per channel
if 'nama_channel' in data.columns:
    sentimen_tiap_channel = pd.pivot_table(
        data, index='nama_channel', columns='Labeling', aggfunc='size', fill_value=0)
    st.bar_chart(sentimen_tiap_channel)

# WordCloud untuk Sentimen Positif, Negatif, dan Netral
data_positif = data[data['Labeling'] == 'positif']
data_negatif = data[data['Labeling'] == 'negatif']
data_netral = data[data['Labeling'] == 'netral']

text_positif = ' '.join(data_positif["clean_text"].dropna().values.tolist())
text_negatif = ' '.join(data_negatif["clean_text"].dropna().values.tolist())
text_netral = ' '.join(data_netral["clean_text"].dropna().values.tolist())

wordcloud_stopword = set(["si", "the", "di", "ini", "aja", "yg", "dan", "yang", "china", "jepang", "mas", "toyota", "and", "k15c", "mirip2", "lihat", "500jt", "400jt", "koq", "vitara", "sih", "tumben", "prindavan", "in", "seater", "deh", "evx", "nih", "ngeriii", "disain2 jelek2", "futuristikmudah2an", "kirakira", "diterusin", "y", "ya", "yak", "nya", "waduhhh", "modal", "mantaapp", "fitur2nya", "templet", "disain2", "jugajangan", "ajaa", "purosaangue", "unggu", "pakai", "konsepanlangsung",
                         "jiplaakkk", "especially", "icgc", "idola", "jelek2", "bagusyahh", "anjayy", "k", "pede", "pabrik2", "ads", "is", "kocak", "fronx", "iya", "gebrak", "kendara", "video", "hdr", "sunat", "tau", "goreng", "mass", "prod", "mass", "eh", "bahagian", "moga", "buat", "beda", "terbang", "bantai", "banding", "piring", "size", "konsap", "mid", "suntik", "mati", "ujung2nya", "ragu", "apung", "sahabat", "pinggir", "massa", "pamer", "main", "cina", "pamerin", "aman", "heran"])


wordcloud_positif = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=wordcloud_stopword,
                              min_font_size=20).generate(text_positif)

wordcloud_negatif = WordCloud(width=800, height=800,
                              background_color='white',
                              stopwords=wordcloud_stopword,
                              min_font_size=20).generate(text_negatif)

wordcloud_netral = WordCloud(width=800, height=800,
                             background_color='white',
                             stopwords=wordcloud_stopword,
                             min_font_size=20).generate(text_netral)

# WordCloud untuk Sentimen Positif
st.subheader("WordCloud - Sentimen Positif")
fig_positif = plt.figure(figsize=(8, 8))
plt.imshow(wordcloud_positif, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig_positif)

# WordCloud untuk Sentimen Netral
st.subheader("WordCloud - Sentimen Netral")
fig_netral = plt.figure(figsize=(8, 8))
plt.imshow(wordcloud_netral, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig_netral)

# WordCloud untuk Sentimen Negatif
st.subheader("WordCloud - Sentimen Negatif")
fig_negatif = plt.figure(figsize=(8, 8))
plt.imshow(wordcloud_negatif, interpolation='bilinear')
plt.axis('off')
st.pyplot(fig_negatif)
