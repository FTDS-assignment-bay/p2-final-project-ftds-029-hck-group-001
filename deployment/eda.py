import streamlit as st
import os
from PIL import Image

# def show():

#     img1 = os.path.join("EDA", "image.png")
#     st.image(img1, caption="Distribusi Tiap Kelas")
#     st.markdown("""
#     **Insights:**
    
#     """)

#     img2 = os.path.join("EDA", "eda2.png")
#     st.image(img2, caption="Distribusi Size Images")
#     st.markdown("""
#     **Insights:**
   
#     """)

#     img3 = os.path.join("EDA", "eda3.png")
#     st.image(img3, caption="Distribusi Color Modes Pada Images")
#     st.markdown("""
#     **Insights:**
    
#     """)

#     # img4 = os.path.join("Deployment", "EDA", "eda4_1.png")
#     # st.image(img4)
#     # img5 = os.path.join("Deployment", "EDA", "eda4_2.png")
#     # st.image(img5)
#     # img6 = os.path.join("Deployment", "EDA", "eda4_3.png")
#     # st.image(img6)
#     # img7 = os.path.join("Deployment", "EDA", "eda4_4.png")
#     # st.image(img7, caption="Sample Tiap Kelas")
#     st.markdown("""
#     **Insights:**
    
#     """)

# if __name__ == "__main__":
#     show()


import streamlit as st
import os
from PIL import Image
import base64
from io import BytesIO

def show():
    # Helper function: center image with HTML/CSS, keeping PIL resize
    def centered_image(img_obj, caption_text):
        buffered = BytesIO()
        img_obj.save(buffered, format="PNG")  # save image to buffer
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        st.markdown(
            f"""
            <div style="text-align: left;">
                <img src="data:image/png;base64,{img_base64}" alt="{caption_text}">
                <p><b>{caption_text}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # First image (resized)
    st.write("## 1. Event Types Distribution")
    img1_path = os.path.join("EDA", "image.png")
    img1 = Image.open(img1_path).resize((1100, 500))
    centered_image(img1, "")
    st.markdown('''**Insights:**\n
1. View mendominasi interaksi dengan 2,66 juta event.
2. Add to Cart jauh lebih sedikit, hanya 69 ribu (±2,6% dari view).
3. Transaction paling kecil, 22 ribu (±0,8% dari view).
4. Pola ini menunjukkan sebagian besar pengguna berhenti di tahap awal interaksi.
                ''')

    # Second image (resized)
    st.write("## 2. Conversion Funnel")
    img2_path = os.path.join("EDA", "eda2.png")
    img2 = Image.open(img2_path).resize((1100, 500))
    centered_image(img2, "")
    st.markdown('''**Insights:**\n
1. Transisi View → Add to Cart: Drop signifikan, mayoritas pengguna tidak melanjutkan ke keranjang.
2. Transisi Add to Cart → Purchase: Hanya ±32% yang melanjutkan ke pembelian.\n
Dua bottleneck utama:
- Minat awal → niat beli (produk/UX kurang menarik).
- Niat beli → transaksi (hambatan checkout, harga, atau kepercayaan).
                ''')

    # Third
    st.write("## 3. Customer Activity")
    img3 = os.path.join("EDA", "eda3.png")
    st.image(img3)
    st.markdown("""
    **Insights:**
1. Aktivitas Berdasarkan Jam:
- Aktivitas tertinggi terjadi pada pukul 17.00–22.00 dan 00.00–04.00.
- Aktivitas terendah ada di pagi hari 07.00–11.00.
- Pola ini menunjukkan pengguna cenderung berinteraksi di malam hingga dini hari, kemungkinan saat waktu luang.
2. Aktivitas Berdasarkan Hari
- Aktivitas harian relatif stabil di hari kerja (Senin–Jumat).
- Terjadi penurunan signifikan pada Sabtu, sedikit meningkat kembali di Minggu.
- Hal ini mengindikasikan interaksi lebih banyak terjadi di hari kerja, mungkin karena akses dilakukan saat jam istirahat atau sela pekerjaan.
    """)


    st.write("# Recommendation")
    st.markdown("""
1. Optimasi Produk & UX untuk Tingkatkan Add to Cart
- Gunakan personalized marketing actions.
- Tambahkan urgency cues (stok terbatas, diskon waktu terbatas) untuk memicu aksi.
2. Timing Promosi Berdasarkan Aktivitas
- Jadwalkan kampanye push notification atau penawaran di jam puncak (17.00–22.00).
- Minimalkan promosi di jam sepi, fokuskan ke jam malam dan dini hari.
3. Strategi Hari Spesifik
- Tingkatkan promosi pada akhir pekan, terutama Sabtu untuk mengimbangi penurunan traffic.
- Manfaatkan RL untuk menguji skenario promo berbeda di tiap hari/jam guna memaksimalkan konversi.

    """)
if __name__ == "__main__":
    show()

