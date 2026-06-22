import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage
)
from reportlab.lib.styles import getSampleStyleSheet

from datetime import datetime

# =====================================
# JUDUL
# =====================================

st.set_page_config(
    page_title="SobatOmpreng",
    page_icon="🍱",
    layout="wide"
)

st.title("🍱 SobatOmpreng")
st.subheader("Dashboard Monitoring Gizi MBG")

# =====================================
# LOAD MODEL
# =====================================

model = YOLO("best.pt")
def buat_pdf(
    makanan,
    kalori,
    protein,
    lemak,
    karbo,
    score,
    kesimpulan
):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    elements = []

    # =====================
    # JUDUL
    # =====================

    elements.append(
        Paragraph(
            "LAPORAN ANALISIS GIZI MBG",
            styles["Title"]
        )
    )

    elements.append(Spacer(1,12))

    tanggal = datetime.now().strftime(
        "%d-%m-%Y %H:%M"
    )

    elements.append(
        Paragraph(
            f"Tanggal Analisis : {tanggal}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1,10))

    # =====================
    # FOTO
    # =====================

    try:

        img = PDFImage(
            "hasil_deteksi.jpg",
            width=350,
            height=250
        )

        elements.append(img)

    except:
        pass

    elements.append(Spacer(1,15))

    # =====================
    # MAKANAN
    # =====================

    elements.append(
        Paragraph(
            "<b>Makanan Terdeteksi</b>",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            ", ".join(makanan),
            styles["Normal"]
        )
    )

    elements.append(Spacer(1,10))

    # =====================
    # NUTRISI
    # =====================

    elements.append(
        Paragraph(
            "<b>Total Nutrisi</b>",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            f"Kalori : {kalori:.1f} kkal",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Protein : {protein:.1f} g",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Lemak : {lemak:.1f} g",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"Karbohidrat : {karbo:.1f} g",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1,10))

    # =====================
    # SCORE
    # =====================

    elements.append(
        Paragraph(
            "<b>SobatOmpreng Score</b>",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            f"{score}/100",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1,10))

    # =====================
    # KESIMPULAN
    # =====================

    elements.append(
        Paragraph(
            "<b>Kesimpulan</b>",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            kesimpulan,
            styles["Normal"]
        )
    )

    doc.build(elements)

    buffer.seek(0)

    return buffer

# =====================================
# LOAD DATA NUTRISI
# =====================================

nutrition = pd.read_csv(
    "nutrition.csv",
    sep=";",
    engine="python"
)

nutrition.columns = [
    "id",
    "calories",
    "proteins",
    "fat",
    "carbohydrate",
    "name",
    "image"
]

# =====================================
# MAPPING KELAS
# =====================================

mapping = {
    "Tahu": "Tahu goreng",
    "Ikan": "Ikan bandeng presto masakan",
    "Telur": "Telur Ayam",
    "Jeruk": "Jeruk Manis",
    "Anggur": "Anggur Hutan Segar",
    "Melon": "Melon segar",
    "Markisa": "Markisa segar",
    "Roti": "Roti Putih",
    "Kripik tempe": "Kripik Tempe Goreng",
    "Petti daging": "Sapi daging kornet",
    "Mie": "Mie Sagu",
    "Jagung": "Jagung Rebus",
    "Daging": "Daging Sapi",
    "Tempe": "Tempe Goreng",
    "Salak": "Salak bali segar",
    "Pepaya": "Pepaya segar",
    "Tomat": "Tomat merah segar",
    "Ayam": "Ayam goreng paha",
    "Cap cai sayur": "Cap cai sayur",
    "Pisang": "Pisang Mas",
    "Kangkung": "Kangkung",
    "Ketimun": "Ketimun",
    "Nasi": "Nasi"
}

# ==========================
# PILIH SUMBER GAMBAR
# ==========================

metode = st.radio(
    "Pilih Sumber Gambar",
    ["Upload Foto", "Kamera"]
)

file = None

# ==========================
# UPLOAD FOTO
# ==========================

if metode == "Upload Foto":

    uploaded_file = st.file_uploader(
        "Upload Foto Ompreng",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file = uploaded_file

# ==========================
# KAMERA
# ==========================

else:

    camera_file = st.camera_input(
        "Ambil Foto Ompreng"
    )

    if camera_file is not None:
        file = camera_file

# ==========================
# PROSES FOTO
# ==========================

if file is not None:

    image = Image.open(file).convert("RGB")

    st.subheader("📷 Foto Ompreng")
    st.image(
        image,
        use_container_width=True
    )

    # ======================
    # DETEKSI YOLO
    # ======================

    results = model.predict(
        image,
        conf=0.10
    )

    # =================================
    # DETEKSI YOLO
    # =================================

    results = model.predict(
        image,
        conf=0.1
    )
    

    img = np.array(image).copy()

    kelas_unik = set()

    total_kalori = 0
    total_protein = 0
    total_lemak = 0
    total_karbo = 0

    tabel_nutrisi = []

    # =================================
    # LOOP DETEKSI
    # =================================

    for box in results[0].boxes:

        cls = int(box.cls[0])

        nama_kelas = model.names[cls]

        # agar tidak double
        if nama_kelas in kelas_unik:
            continue

        kelas_unik.add(nama_kelas)

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        nama_dataset = mapping.get(
            nama_kelas,
            nama_kelas
        )

        try:

            row = nutrition[
                nutrition["name"]
                .astype(str)
                .str.lower()
                ==
                nama_dataset.lower()
            ].iloc[0]

            kalori = float(row["calories"])
            protein = float(row["proteins"])
            lemak = float(row["fat"])
            karbo = float(row["carbohydrate"])

            total_kalori += kalori
            total_protein += protein
            total_lemak += lemak
            total_karbo += karbo

            tabel_nutrisi.append({
                "Makanan": nama_kelas,
                "Kalori": kalori,
                "Protein": protein,
                "Lemak": lemak,
                "Karbohidrat": karbo
            })

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                2
            )

            cv2.putText(
                img,
                nama_kelas,
                (x1, max(y1-10,20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        except:
            pass

    # =================================
    # HASIL DETEKSI
    # =================================

    st.subheader("🔍 Hasil Deteksi")

    st.image(
        img,
        use_container_width=True
    )

    cv2.imwrite(
        "hasil_deteksi.jpg",
    cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )
)

    # =================================
    # MAKANAN TERDETEKSI
    # =================================

    st.subheader("🍽️ Makanan Terdeteksi")

    st.write(
        ", ".join(sorted(kelas_unik))
    )

    # =================================
    # TABEL NUTRISI
    # =================================

    st.subheader("📋 Nutrisi Per Makanan")

    if len(tabel_nutrisi) > 0:

        st.dataframe(
            pd.DataFrame(
                tabel_nutrisi
            ),
            use_container_width=True
        )

    # =================================
    # TOTAL NUTRISI
    # =================================

    st.subheader("📊 Total Nutrisi Ompreng")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Kalori",
        f"{total_kalori:.1f}"
    )

    c2.metric(
        "Protein",
        f"{total_protein:.1f} g"
    )

    c3.metric(
        "Lemak",
        f"{total_lemak:.1f} g"
    )

    c4.metric(
        "Karbohidrat",
        f"{total_karbo:.1f} g"
    )

    # =================================
    # AKG SEKALI MAKAN
    # =================================

    AKG_KALORI = 500
    AKG_PROTEIN = 20
    AKG_LEMAK = 15
    AKG_KARBO = 60

    

    st.subheader("📈 Progress AKG")

    persen_kalori = min(
    total_kalori / AKG_KALORI,
    1.0
)

    persen_protein = min(
    total_protein / AKG_PROTEIN,
    1.0
)

    persen_lemak = min(
    total_lemak / AKG_LEMAK,
    1.0
)

    persen_karbo = min(
    total_karbo / AKG_KARBO,
    1.0
)

    st.write(
    f"Kalori ({persen_kalori*100:.0f}%)"
)
    st.progress(persen_kalori)

    st.write(
    f"Protein ({persen_protein*100:.0f}%)"
)
    st.progress(persen_protein)

    st.write(
    f"Lemak ({persen_lemak*100:.0f}%)"
)
    st.progress(persen_lemak)

    st.write(
    f"Karbohidrat ({persen_karbo*100:.0f}%)"
)
    st.progress(persen_karbo)
    st.subheader("🎯 Evaluasi AKG Sekali Makan")

    st.write(
        f"Kalori : {'✅ Memenuhi' if total_kalori >= AKG_KALORI else '❌ Belum Memenuhi'}"
    )

    st.write(
        f"Protein : {'✅ Memenuhi' if total_protein >= AKG_PROTEIN else '❌ Belum Memenuhi'}"
    )

    st.write(
        f"Lemak : {'✅ Memenuhi' if total_lemak >= AKG_LEMAK else '❌ Belum Memenuhi'}"
    )

    st.write(
        f"Karbohidrat : {'✅ Memenuhi' if total_karbo >= AKG_KARBO else '❌ Belum Memenuhi'}"
    ) # =================================
    # KESIMPULAN
    # =================================

    st.subheader("📝 Kesimpulan")

    if (
        total_kalori >= AKG_KALORI
        and total_protein >= AKG_PROTEIN
        and total_lemak >= AKG_LEMAK
        and total_karbo >= AKG_KARBO
):

        kesimpulan_pdf = (
            "Menu memenuhi kebutuhan gizi sekali makan"
    )

        st.success(kesimpulan_pdf)

    else:

        kesimpulan_pdf = (
            "Menu belum memenuhi kebutuhan gizi sekali makan"
    )

        st.warning(kesimpulan_pdf)

# ======================
# STATUS KELAYAKAN
# ======================

    st.subheader("🏆 Status Kelayakan Menu")

    score = 0

    if total_kalori >= AKG_KALORI:
        score += 25

    if total_protein >= AKG_PROTEIN:
        score += 25

    if total_lemak >= AKG_LEMAK:
        score += 25

    if total_karbo >= AKG_KARBO:
        score += 25

    st.metric(
    "SobatOmpreng Score",
    f"{score}/100"
)

    if score >= 80:
        st.success("🟢 Menu Layak")
    elif score >= 60:
        st.warning("🟡 Menu Cukup Layak")
    else:
        st.error("🔴 Menu Tidak Layak")

# ======================
# GRAFIK NUTRISI
# ======================

    st.subheader("📈 Komposisi Nutrisi")

    fig, ax = plt.subplots()

    ax.pie(
    [
        total_protein,
        total_lemak,
        total_karbo,
        total_kalori
    ],
    labels=[
        "Protein",
        "Lemak",
        "Karbohidrat",
        "Kalori"
    ],
    autopct="%1.1f%%"
)

    st.pyplot(fig)

# ======================
# DOWNLOAD PDF
# ======================

    st.subheader("📄 Laporan Analisis")

    pdf_file = buat_pdf(
        list(kelas_unik),
        total_kalori,
        total_protein,
        total_lemak,
        total_karbo,
        score,
        kesimpulan_pdf
)

    st.download_button(
        label="⬇️ Download Laporan PDF",
        data=pdf_file,
        file_name="Laporan_SobatOmpreng.pdf",
        mime="application/pdf"
)