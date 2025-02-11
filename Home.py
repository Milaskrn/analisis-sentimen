import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Project Analisis Sentimen",
    page_icon="📚",
)

# Judul di tengah dan melebar
st.markdown(
    """
    <div style='width: 100%; text-align: center; font-size: 35px; font-weight: bold;'>
        Sentimen Analisis Ketertarikan Pembelian Mobil Listrik Suzuki Menggunakan Metode Klasifikasi Logistic Regression
    </div>
    """,
    unsafe_allow_html=True
)


# Memuat logo Suzuki
logo = Image.open("suzuki.png")  # Ganti dengan path logo yang sesuai

# Menampilkan logo dan teks di bagian atas sidebar
with st.sidebar:
    st.image(logo, use_container_width=True)  # Menampilkan logo di atas
    st.markdown(
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

# Menampilkan video YouTube di tengah
st.markdown(
    """
    <div style="justify-content: center;">
        <iframe width="700" height="600" src="https://www.youtube.com/embed/q1QA0VVSDwY?si=SO_68GHUj72-hIrm" frameborder="0" allowfullscreen></iframe>
    </div>
    """,
    unsafe_allow_html=True
)

# Menampilkan teks konten penelitian di bawah video
st.markdown(
    """
    <div style='text-align: justify; font-size: 20px; margin-top: 30px; width: 100%;'>
        <b>Abstrak:</b> 
        Penelitian ini bertujuan untuk menganalisis sentimen masyarakat terhadap ketertarikan pembelian mobil listrik Suzuki 
        dengan menggunakan metode klasifikasi Logistic Regression. Dengan meningkatnya minat terhadap kendaraan listrik, 
        pemahaman terhadap opini publik menjadi penting untuk strategi pemasaran dan pengembangan produk. 
        Dataset yang digunakan berasal dari berbagai ulasan pengguna di media sosial dan platform e-commerce. 
        Hasil penelitian ini diharapkan dapat memberikan wawasan mengenai faktor-faktor yang mempengaruhi minat masyarakat 
        terhadap mobil listrik serta membantu Suzuki dalam merancang kebijakan yang lebih tepat.
    </div>
    """,
    unsafe_allow_html=True
)
