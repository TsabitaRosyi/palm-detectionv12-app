import streamlit as st
from PIL import Image
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
from supervision import BoxAnnotator, LabelAnnotator, Color, Detections
from io import BytesIO
import base64
import tempfile
import plotly.express as px


# =============================
# Fungsi konversi gambar → base64
# =============================
def image_to_base64(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# =============================
# Konfigurasi halaman
# =============================
st.set_page_config(page_title="Deteksi Buah Sawit", layout="wide")


# =============================
# Load Model YOLO
# =============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()


# =============================
# Warna label
# =============================
label_to_color = {
    "matang": Color.RED,
    "mengkal": Color.YELLOW,
    "mentah": Color.BLACK
}


# =============================
# Fungsi anotasi YOLO
# =============================
def draw_results(image, results, text_scale=1.5):

    img = np.array(image.convert("RGB"))
    class_counts = Counter()

    label_annotator = LabelAnnotator(
        text_scale=text_scale,
        text_thickness=2,
        text_padding=5
    )

    for result in results:

        boxes = result.boxes
        names = result.names

        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(xyxy, class_ids, confidences):

            if class_id not in names:
                continue

            class_name = names[class_id].strip().lower()
            label = f"{class_name}: {conf:.2f}"

            color = label_to_color.get(class_name, Color.WHITE)
            class_counts[class_name] += 1

            box_annotator = BoxAnnotator(
                color=color,
                thickness=5
            )

            detection = Detections(
                xyxy=np.array([box]),
                confidence=np.array([conf]),
                class_id=np.array([class_id])
            )

            img = box_annotator.annotate(
                scene=img,
                detections=detection
            )

            img = label_annotator.annotate(
                scene=img,
                detections=detection,
                labels=[label]
            )

    return Image.fromarray(img), class_counts


# =============================
# Fungsi crop foto
# =============================
def crop_center_square(img):
    width, height = img.size
    min_dim = min(width, height)

    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2

    return img.crop((left, top, right, bottom))


# =============================
# Load foto profil
# =============================
profile_img = Image.open("foto.jpg")
profile_img = crop_center_square(profile_img)


# =============================
# Sidebar
# =============================
with st.sidebar:

    st.image("logo.png", width=150)

    st.markdown("<h4>Pilih metode input:</h4>", unsafe_allow_html=True)

    option = st.radio(
        "",
        ["Upload Gambar", "Upload Video"],
        label_visibility="collapsed"
    )

    st.markdown(
        f"""
        <style>
        .created-by-container {{
            display:flex;
            align-items:center;
            gap:10px;
            margin-top:20px;
            padding-top:10px;
            border-top:1px solid #ccc;
        }}
        .created-by-img {{
            width:45px;
            height:45px;
            border-radius:50%;
            border:2px solid #444;
            object-fit:cover;
        }}
        .created-by-text {{
            font-size:14px;
            color:#555;
            font-style:italic;
        }}
        </style>

        <div class="created-by-container">
            <img class="created-by-img"
            src="data:image/png;base64,{image_to_base64(profile_img)}" />
            <div class="created-by-text">
            Created by : Tsabit
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =============================
# Judul Halaman
# =============================
st.markdown(
"<h1 style='text-align:center;'>🌴 Deteksi Kematangan Buah Sawit</h1>",
unsafe_allow_html=True
)

st.markdown("""
<div style="text-align:center; font-size:16px; max-width:800px; margin:auto;">
Sistem ini menggunakan teknologi YOLOv12 untuk mendeteksi
kematangan buah kelapa sawit secara otomatis
berdasarkan gambar atau video input.
</div>
""", unsafe_allow_html=True)


# ==========================================================
# MODE GAMBAR
# ==========================================================
if option == "Upload Gambar":

    uploaded_file = st.file_uploader(
        "Unggah Gambar",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        with st.spinner("🔍 Memproses gambar..."):
            results = model(image)
            result_img, class_counts = draw_results(
                image,
                results,
                text_scale=1.5
            )

        col_input, col_output = st.columns(2)

        with col_input:
            st.image(image, use_container_width=True)

        with col_output:
            st.image(result_img, use_container_width=True)

        buf = BytesIO()
        result_img.save(buf, format="PNG")

        st.download_button(
            "⬇️ Download Hasil Deteksi",
            buf.getvalue(),
            "hasil_deteksi.png",
            "image/png"
        )

        mentah = class_counts.get("mentah",0)
        mengkal = class_counts.get("mengkal",0)
        matang = class_counts.get("matang",0)

        data_chart = {
            "Kategori":["Mentah","Mengkal","Matang"],
            "Jumlah":[mentah,mengkal,matang]
        }

        fig = px.pie(
            data_chart,
            names="Kategori",
            values="Jumlah",
            hole=0.3
        )

        st.plotly_chart(fig,use_container_width=True)


# ==========================================================
# MODE VIDEO
# ==========================================================
elif option == "Upload Video":

    uploaded_video = st.file_uploader(
        "Unggah Video",
        type=["mp4","avi","mov"]
    )

    if uploaded_video:

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        output_path = "hasil_deteksi_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path,fourcc,fps,(width,height))

        stframe = st.empty()

        with st.spinner("🔍 Memproses video..."):

            while cap.isOpened():

                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame)

                annotated_img,_ = draw_results(
                    Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ),
                    results,
                    text_scale=0.8
                )

                annotated_bgr = cv2.cvtColor(
                    np.array(annotated_img),
                    cv2.COLOR_RGB2BGR
                )

                out.write(annotated_bgr)

                stframe.image(
                    annotated_bgr,
                    channels="BGR",
                    use_container_width=True
                )

        cap.release()
        out.release()

        st.success("✅ Video selesai diproses!")

        with open(output_path,"rb") as f:
            st.download_button(
                "⬇️ Download Video Hasil Deteksi",
                f,
                "hasil_deteksi_sawit.mp4",
                "video/mp4"
            )
