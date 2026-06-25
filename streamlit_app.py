import io
import os
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import fitz
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pymongo import MongoClient
from pymongo.collection import Collection
from dotenv import load_dotenv

load_dotenv(override=True)


CLASSES = ["MYTHOLOGY", "LAND", "LITERATURE", "SCIENCE", "GEOGRAPHY", "BANGRAGY","GK"]

DEFAULT_DB = "bengali_words"
DEFAULT_URI = "mongodb://127.0.0.1:27017"

@dataclass
class WordCrop:
    index: int
    image: np.ndarray
    bbox: Tuple[int, int, int, int]
    source: str = "auto"


@st.cache_resource
def get_db():
    uri = os.environ.get("MONGODB_URI")
    db_name = os.environ.get("MONGODB_DB_NAME", DEFAULT_DB)
    client = MongoClient(uri)
    return client[db_name]


def get_collection() -> Collection:
    return get_db()["paragraphs"]


def ensure_indexes():
    col = get_collection()
    col.create_index([("class_name", 1), ("serial_no", 1)], unique=True)


def tokenize_bengali(text: str) -> List[str]:
    tokens = [w.strip() for w in text.replace("\n", " ").split(" ") if w.strip()]
    return tokens


def get_next_serial(class_name: str) -> int:
    col = get_collection()
    doc = col.find_one({"class_name": class_name}, sort=[("serial_no", -1)])
    return 1 if not doc else int(doc["serial_no"]) + 1


def fetch_class_docs(class_name: str) -> List[Dict]:
    col = get_collection()
    return list(col.find({"class_name": class_name}).sort("serial_no", 1))


def add_paragraph(class_name: str, text: str):
    words = tokenize_bengali(text)
    if not words:
        raise ValueError("Text does not contain valid words.")
    serial = get_next_serial(class_name)
    doc = {
        "class_name": class_name,
        "serial_no": serial,
        "text": text.strip(),
        "words": words,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    get_collection().insert_one(doc)


def update_paragraph(doc_id, class_name: str, text: str):
    words = tokenize_bengali(text)
    if not words:
        raise ValueError("Text does not contain valid words.")
    get_collection().update_one(
        {"_id": doc_id},
        {
            "$set": {
                "class_name": class_name,
                "text": text.strip(),
                "words": words,
                "updated_at": datetime.utcnow(),
            }
        },
    )


def delete_paragraph(doc_id):
    get_collection().delete_one({"_id": doc_id})


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_png_bytes(img: np.ndarray) -> bytes:
    ok, buff = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("Could not encode image.")
    return buff.tobytes()


def segment_word_images(image: Image.Image, kx: int = 15, ky: int = 3) -> List[WordCrop]:
    img = pil_to_cv2(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoise = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Background normalization: removes gray paper background and white gaps
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    bg = cv2.dilate(denoise, kernel_bg)
    diff = cv2.absdiff(bg, denoise)
    
    _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h_img, w_img = gray.shape[:2]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    # Use user-provided horizontal (kx) and vertical (ky) connection gaps
    connect = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)), iterations=1)

    contours, _ = cv2.findContours(connect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    boxes = []
    # Estimate single page area to prevent min_area scaling wildly on multipage docs
    single_page_area = w_img * w_img * 1.414
    min_area = max(60, int(single_page_area * 0.00008))
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        if h < 8 or w < 8:
            continue
        boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: (b[1] // 25, b[0]))
    crops: List[WordCrop] = []

    for i, (x, y, w, h) in enumerate(boxes):
        pad = 4
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        crop = img[y1:y2, x1:x2]
        crops.append(WordCrop(index=i, image=crop, bbox=(x1, y1, x2 - x1, y2 - y1), source="auto"))
    return crops


def init_mapping_state():
    if "mapping_rows" not in st.session_state:
        st.session_state.mapping_rows = []
    if "manual_counter" not in st.session_state:
        st.session_state.manual_counter = 0


def reset_mapping():
    st.session_state.mapping_rows = []
    st.session_state.manual_counter = 0


def build_export(mapping_rows: List[Dict]) -> Tuple[bytes, bytes]:
    df_rows = []
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, row in enumerate(mapping_rows, start=1):
            file_name = f"word_{i:03d}.png"
            img_bytes = row["img_bytes"]
            zf.writestr(file_name, img_bytes)
            df_rows.append(
                {
                    "position": i,
                    "digital_word": row["digital_word"],
                    "image_file": file_name,
                    "source": row["source"],
                }
            )

    excel_buffer = io.BytesIO()
    pd.DataFrame(df_rows).to_excel(excel_buffer, index=False)
    return excel_buffer.getvalue(), zip_buffer.getvalue()


def developer_ui():
    st.header("Developer UI")
    selected_class = st.selectbox("Select Text Class", CLASSES, key="dev_class")
    input_text = st.text_area("Enter Digital Bengali Paragraph Text", height=140, key="dev_input_text")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Save Paragraph", use_container_width=True):
            try:
                add_paragraph(selected_class, input_text)
                st.success("Paragraph saved.")
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")
    with c2:
        if st.button("Clear Input", use_container_width=True):
            st.session_state.dev_input_text = ""
            st.rerun()

    st.subheader("Existing Text Entries")
    docs = fetch_class_docs(selected_class)
    if not docs:
        st.info("No entry found for this class.")
        return

    for doc in docs:
        with st.expander(f"Serial {doc['serial_no']} | {doc['class_name']}", expanded=False):
            new_class = st.selectbox(
                "Class",
                CLASSES,
                index=CLASSES.index(doc["class_name"]),
                key=f"class_{doc['_id']}",
            )
            new_text = st.text_area("Text", value=doc["text"], height=120, key=f"text_{doc['_id']}")
            words_preview = tokenize_bengali(new_text)
            st.caption(f"Word Count: {len(words_preview)}")
            st.write(words_preview)

            ucol, dcol = st.columns([1, 1])
            with ucol:
                if st.button("Update", key=f"update_{doc['_id']}", use_container_width=True):
                    try:
                        update_paragraph(doc["_id"], new_class, new_text)
                        st.success("Updated.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Update failed: {e}")
            with dcol:
                if st.button("Delete", key=f"delete_{doc['_id']}", use_container_width=True):
                    delete_paragraph(doc["_id"])
                    st.warning("Deleted.")
                    st.rerun()


def user_ui():
    st.header("User UI")
    init_mapping_state()

    selected_class = st.selectbox("Select Class", CLASSES, key="user_class")
    docs = fetch_class_docs(selected_class)
    if not docs:
        st.info("No text found in this class. Ask developer to add text first.")
        return

    options = {f"Serial {d['serial_no']}": d for d in docs}
    selected_key = st.selectbox("Select Text (Serial Number)", list(options.keys()), key="user_serial")
    selected_doc = options[selected_key]

    st.text_area("Selected Digital Paragraph", selected_doc["text"], disabled=True, height=120)
    st.caption(f"Digital Words: {len(selected_doc['words'])}")

    uploaded = st.file_uploader("Upload Handwritten Paragraph (Image or PDF)", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded is not None:
        if uploaded.name.lower().endswith('.pdf'):
            doc = fitz.open(stream=uploaded.read(), filetype="pdf")
            mat = fitz.Matrix(2.5, 2.5)
            images = []
            for page in doc:
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            
            if images:
                total_height = sum(img.height for img in images)
                max_width = max(img.width for img in images)
                concat_img = Image.new('RGB', (max_width, total_height), (255, 255, 255))
                y_offset = 0
                for img in images:
                    concat_img.paste(img, (0, y_offset))
                    y_offset += img.height
                image = concat_img
            else:
                st.error("Uploaded PDF contains no pages.")
                return
        else:
            image = Image.open(uploaded)

        st.image(image, caption="Uploaded Paragraph", use_container_width=True)

        st.subheader("Segmentation Adjustments")
        col_kx, col_ky = st.columns(2)
        with col_kx:
            word_gap = st.slider("Horizontal Word Gap (px)", min_value=1, max_value=150, value=15, help="Increase to merge characters. Decrease if separate words are merging.")
        with col_ky:
            line_gap = st.slider("Vertical Line Gap (px)", min_value=1, max_value=50, value=3, help="Increase if words are split vertically. Decrease if separate lines are merging.")

        if st.button("Auto Generate Word Images"):
            crops = segment_word_images(image, kx=word_gap, ky=line_gap)
            rows = []
            digital_words = selected_doc["words"]
            for i, crop in enumerate(crops):
                mapped_word = digital_words[i] if i < len(digital_words) else ""
                rows.append(
                    {
                        "id": f"auto_{i}",
                        "digital_word": mapped_word,
                        "img_bytes": cv2_to_png_bytes(crop.image),
                        "source": crop.source,
                    }
                )
            st.session_state.mapping_rows = rows
            st.success(f"Generated {len(rows)} word images.")

    if st.session_state.mapping_rows:
        st.subheader("Map Table (Image -> Digital Word)")
        rows = st.session_state.mapping_rows

        for i in range(len(rows)):
            row = rows[i]
            c1, c2, c3, c4, c5, c6, c7 = st.columns([2, 2, 0.8, 0.8, 1.2, 1.2, 1.2])
            with c1:
                st.image(row["img_bytes"], caption=f"#{i+1}", use_container_width=False)
            with c2:
                row["digital_word"] = st.text_input(
                    f"Word {i+1}", value=row["digital_word"], key=f"word_map_{row['id']}"
                )
                st.caption(f"Source: {row['source']}")
            with c3:
                if st.button("Up", key=f"up_{row['id']}", use_container_width=True) and i > 0:
                    rows[i - 1], rows[i] = rows[i], rows[i - 1]
                    st.rerun()
            with c4:
                if st.button("Down", key=f"down_{row['id']}", use_container_width=True) and i < len(rows) - 1:
                    rows[i + 1], rows[i] = rows[i], rows[i + 1]
                    st.rerun()
            with c5:
                if st.button("Del Img", key=f"del_img_{row['id']}", use_container_width=True):
                    for j in range(i, len(rows) - 1):
                        rows[j]["img_bytes"] = rows[j+1]["img_bytes"]
                        rows[j]["source"] = rows[j+1]["source"]
                    
                    last_word_key = f"word_map_{rows[-1]['id']}"
                    last_word = st.session_state.get(last_word_key, rows[-1]["digital_word"])
                    if not last_word.strip():
                        rows.pop()
                    else:
                        blank = np.zeros((10, 10, 3), dtype=np.uint8) + 255
                        ok, buff = cv2.imencode(".png", blank)
                        rows[-1]["img_bytes"] = buff.tobytes()
                        rows[-1]["source"] = "blank"
                    st.rerun()
            with c6:
                if st.button("Del Word", key=f"del_word_{row['id']}", use_container_width=True):
                    for j in range(i, len(rows) - 1):
                        next_key = f"word_map_{rows[j+1]['id']}"
                        next_word = st.session_state.get(next_key, rows[j+1]["digital_word"])
                        rows[j]["digital_word"] = next_word
                        curr_key = f"word_map_{rows[j]['id']}"
                        st.session_state[curr_key] = next_word
                    rows[-1]["digital_word"] = ""
                    last_key = f"word_map_{rows[-1]['id']}"
                    st.session_state[last_key] = ""
                    st.rerun()
            with c7:
                if st.button("Del Both", key=f"del_both_{row['id']}", use_container_width=True):
                    rows.pop(i)
                    st.rerun()

        st.divider()
        st.subheader("Manual Word Image Upload")
        manual_img = st.file_uploader(
            "Add one manual word image", type=["png", "jpg", "jpeg"], key="manual_uploader"
        )
        manual_word = st.text_input("Digital word for manual image", key="manual_word")
        if st.button("Add Manual Mapping"):
            if manual_img is None:
                st.error("Upload an image first.")
            else:
                img = Image.open(manual_img).convert("RGB")
                img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                st.session_state.manual_counter += 1
                rows.append(
                    {
                        "id": f"manual_{st.session_state.manual_counter}",
                        "digital_word": manual_word.strip(),
                        "img_bytes": cv2_to_png_bytes(img_np),
                        "source": "manual",
                    }
                )
                st.success("Manual mapping added.")
                st.rerun()

        st.divider()
        if st.button("Reset Current Mapping"):
            reset_mapping()
            st.rerun()

        excel_bytes, zip_bytes = build_export(rows)
        colx, colz = st.columns(2)
        with colx:
            st.download_button(
                "Download Excel Map",
                data=excel_bytes,
                file_name="word_mapping.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with colz:
            st.download_button(
                "Download Word Images ZIP",
                data=zip_bytes,
                file_name="word_images.zip",
                mime="application/zip",
                use_container_width=True,
            )


def main():
    st.set_page_config(page_title="Bengali Handwritten Word Splitter", layout="wide")
    ensure_indexes()
    st.title("Bengali Handwritten Word Splitter")
    mode = st.radio("Choose Mode", ["Developer", "User"], horizontal=True)
    if mode == "Developer":
        developer_ui()
    else:
        user_ui()


if __name__ == "__main__":
    main()
