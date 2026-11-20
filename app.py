import streamlit as st
import cv2
import numpy as np
import fitz
import pandas as pd
import json
import os
import re
import io

st.set_page_config(page_title="Corrector Cambridge B1–C2", layout="wide")

DB_FILE = "banco_examenes_b1_c2.json"

def load_exams():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "Demo B2 Listening Oficial (Test 1)": {
            "plantilla": "Plantilla 1: Listening Oficial (B1–C2)",
            "nivel": "B2",
            "respuestas": {
                "1": "B", "2": "A", "3": "C", "4": "B", "5": "A", "6": "C", "7": "B", "8": "A",
                "9": "GARDENER", "10": "1985", "11": "GREENHOUSE", "12": "SOIL", "13": "TOOLS",
                "14": "SUNLIGHT", "15": "WATERING", "16": "SEEDS", "17": "AUTUMN", "18": "MARKET",
                "19": "E", "20": "A", "21": "H", "22": "C", "23": "F",
                "24": "B", "25": "A", "26": "C", "27": "B", "28": "A", "29": "C", "30": "B"
            }
        },
        "Demo C1 Use of English + Reading (Sample)": {
            "plantilla": "Plantilla 2: Reading & Use of English (B1–C2)",
            "nivel": "C1",
            "respuestas": {
                "1": "B", "2": "D", "3": "A", "4": "C", "5": "B", "6": "A", "7": "D", "8": "C",
                "9": "WHO", "10": "DESPITE", "11": "BEEN", "12": "SUCH",
                "17": "AWARENESS", "18": "SIGNIFICANTLY",
                "31": "B", "32": "A", "33": "D", "34": "C",
                "47": "C", "48": "A", "49": "D", "50": "B"
            }
        },
        "Demo Test Generator (Units 1-4)": {
            "plantilla": "Plantilla 3: Test Generator Doble Cara (EX 1–12)",
            "nivel": "B2",
            "respuestas": {
                "EX1-1": "B", "EX1-2": "A", "EX1-3": "D", "EX1-4": "C",
                "EX2-1": "HAVE BEEN WORKING", "EX2-2": "WOULD HAVE GONE", "EX2-3": "DESPITE",
                "EX7-1": "C", "EX7-2": "A", "EX7-3": "B"
            }
        }
    }

def save_exams(exams):
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(exams, f, ensure_ascii=False, indent=2)

def order_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[int(np.argmin(s))]
    tr = pts[int(np.argmin(diff))]
    br = pts[int(np.argmax(s))]
    bl = pts[int(np.argmax(diff))]
    return np.array((tl, tr, br, bl), dtype="float32")

def four_point_transform(image, pts):
    rect = order_points(pts)
    width, height = 1200, 1740
    dst = np.array(
        ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)),
        dtype="float32"
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (width, height))

def align_sheet(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_img, w_img = gray.shape
    min_area = (w_img * h_img) * 0.0002
    max_area = (w_img * h_img) * 0.0065
    fiducials = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.70 <= aspect_ratio <= 1.40:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        fiducials.append((cx, cy))
    if len(fiducials) < 4:
        return None, "No se detectaron las 4 marcas negras de las esquinas. Asegúrate de que salgan enteras en la foto."
    pts = np.array(fiducials, dtype="float32")
    corners = order_points(pts)
    return four_point_transform(img_bgr, corners), None

def read_omr_row(warped_thresh, x_start, x_step, cy, num_opts):
    letters = ("A", "B", "C", "D", "E", "F", "G", "H")
    fill_ratios = []
    coords = []
    for opt_idx in range(num_opts):
        cx = int(x_start + opt_idx * x_step)
        x1, y1, x2, y2 = cx - 11, cy - 8, cx + 11, cy + 8
        roi = warped_thresh[y1:y2, x1:x2]
        ratio = cv2.countNonZero(roi) / float(roi.size) if roi.size > 0 else 0.0
        fill_ratios.append(ratio)
        coords.append((x1, y1, x2, y2))
    max_fill = max(fill_ratios)
    chosen_idx = int(np.argmax(fill_ratios))
    chosen = letters[chosen_idx] if max_fill > 0.38 else "-"
    return chosen, chosen_idx, coords

def draw_box(img, box, color, thickness=3):
    cv2.rectangle(img, (box, box), (box, box), color, thickness)

def process_listening(img_bgr, answer_key):
    warped, err = align_sheet(img_bgr)
    if err:
        return None, None, None, err
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    header = warped[105:245, 30:1170]
    annotated = warped.copy()
    omr_results = {}
    written_crops = {}
    letters = ("A", "B", "C", "D", "E", "F", "G", "H")

    for idx, q in enumerate(range(1, 9)):
        cy = int(385 + idx * 44.5)
        chosen, c_idx, coords = read_omr_row(thresh, 145, 42.0, cy, 3)
        omr_results[str(q)] = chosen
        if str(q) in answer_key and answer_key[str(q)] in letters[:3]:
            corr = answer_key[str(q)]
            k_idx = letters.index(corr)
            draw_box(annotated, coords[k_idx], (0, 180, 0), 3)
            if chosen != corr and chosen != "-":
                draw_box(annotated, coords[c_idx], (0, 0, 220), 3)

    for idx, q in enumerate(range(7, 19)):
        y1 = int(380 + idx * 45.0)
        written_crops[str(q)] = warped[y1:y1 + 38, 665:1145]

    for idx, q in enumerate(range(14, 24)):
        cy = int(1020 + idx * 45.0)
        chosen, c_idx, coords = read_omr_row(thresh, 145, 39.0, cy, 8)
        omr_results[str(q)] = chosen
        if str(q) in answer_key and answer_key[str(q)] in letters:
            corr = answer_key[str(q)]
            k_idx = letters.index(corr)
            draw_box(annotated, coords[k_idx], (0, 180, 0), 3)
            if chosen != corr and chosen != "-":
                draw_box(annotated, coords[c_idx], (0, 0, 220), 3)

    for idx, q in enumerate(range(20, 31)):
        cy = int(1020 + idx * 45.0)
        chosen, c_idx, coords = read_omr_row(thresh, 710, 39.0, cy, 8)
        omr_results[str(q)] = chosen
        if str(q) in answer_key and answer_key[str(q)] in letters:
            corr = answer_key[str(q)]
            k_idx = letters.index(corr)
            draw_box(annotated, coords[k_idx], (0, 180, 0), 3)
            if chosen != corr and chosen != "-":
                draw_box(annotated, coords[c_idx], (0, 0, 220), 3)

    return omr_results, written_crops, (header, annotated), None

def process_reading_uoe(img_bgr, answer_key):
    warped, err = align_sheet(img_bgr)
    if err:
        return None, None, None, err
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    header = warped[105:245, 30:1170]
    annotated = warped.copy()
    omr_results = {}
    written_crops = {}
    letters = ("A", "B", "C", "D", "E", "F", "G", "H")

    for idx, q in enumerate(range(1, 9)):
        cy = int(385 + idx * 44.0)
        chosen, c_idx, coords = read_omr_row(thresh, 115, 38.0, cy, 4)
        omr_results[str(q)] = chosen
        if str(q) in answer_key and answer_key[str(q)] in letters[:4]:
            corr = answer_key[str(q)]
            k_idx = letters.index(corr)
            draw_box(annotated, coords[k_idx], (0, 180, 0), 3)
            if chosen != corr and chosen != "-":
                draw_box(annotated, coords[c_idx], (0, 0, 220), 3)

    for idx, q in enumerate(range(9, 31)):
        y1 = int(760 + idx * 41.5)
        written_crops[str(q)] = warped[y1:y1 + 36, 110:640]

    for idx, q in enumerate(range(31, 57)):
        cy = int(385 + idx * 48.0)
        chosen, c_idx, coords = read_omr_row(thresh, 745, 38.0, cy, 8)
        omr_results[str(q)] = chosen
        if str(q) in answer_key and answer_key[str(q)] in letters:
            corr = answer_key[str(q)]
            k_idx = letters.index(corr)
            draw_box(annotated, coords[k_idx], (0, 180, 0), 3)
            if chosen != corr and chosen != "-":
                draw_box(annotated, coords[c_idx], (0, 0, 220), 3)

    return omr_results, written_crops, (header, annotated), None

def process_test_generator(img_bgr, answer_key, side=1):
    warped, err = align_sheet(img_bgr)
    if err:
        return None, None, None, err
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    header = warped[105:245, 30:1170]
    annotated = warped.copy()
    omr_results = {}
    written_crops = {}
    letters = ("A", "B", "C", "D")
    start_ex = 1 if side == 1 else 7
    col_x = (115, 650)
    row_y_blocks = (355, 815, 1275)

    for b_idx in range(6):
        ex_num = start_ex + b_idx
        c_idx = b_idx % 2
        r_block = b_idx // 2
        base_x = col_x[c_idx]
        base_y = row_y_blocks[r_block]
        for q_i in range(1, 13):
            key_id = f"EX{ex_num}-{q_i}"
            cy = int(base_y + (q_i - 1) * 32.5)
            chosen, ch_idx, coords = read_omr_row(thresh, base_x, 26.0, cy, 4)
            omr_results[key_id] = chosen
            written_crops[key_id] = warped[cy - 13:cy + 13, base_x + 115:base_x + 445]
            if key_id in answer_key and answer_key[key_id] in letters:
                corr = answer_key[key_id]
                k_idx = letters.index(corr)
                draw_box(annotated, coords[k_idx], (0, 180, 0), 3)
                if chosen != corr and chosen != "-":
                    draw_box(annotated, coords[ch_idx], (0, 0, 220), 3)

    return omr_results, written_crops, (header, annotated), None

def parse_pdf_key(pdf_bytes, plantilla):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join([p.get_text() for p in doc])
    key = {}
    if "Plantilla 3" in plantilla:
        ex_curr = 1
        for line in text.splitlines():
            ex_m = re.search(r'(?:Exercise|Ex|Part)\s*(\d+)', line, re.IGNORECASE)
            if ex_m:
                ex_curr = int(ex_m.group(1))
            q_m = re.match(r'^\s*(\d{1,2})\s*[\.\)\:\-]\s*(.+)$', line.strip())
            if q_m:
                q_n = int(q_m.group(1))
                if 1 <= q_n <= 12 and 1 <= ex_curr <= 12:
                    key[f"EX{ex_curr}-{q_n}"] = q_m.group(2).strip().upper()
    else:
        for line in text.splitlines():
            m = re.match(r'^\s*(\d{1,2})\s*[\.\)\:\-]\s*(.+)$', line.strip())
            if m:
                key[m.group(1)] = m.group(2).strip().upper()
    return key

st.title("Corrector Cambridge B1–C2 — Academia")
st.caption("Soporte completo para Listening Oficial, Reading & Use of English Oficial y Test Generator (Doble Cara).")

if "exams" not in st.session_state:
    st.session_state.exams = load_exams()

tab1, tab2 = st.tabs(("1. Corregir Exámenes", "2. Cargar Examen / Test Generator"))

with tab2:
    st.subheader("Nuevo examen en el banco compartido")
    c1, c2, c3 = st.columns((2, 1, 2))
    with c1:
        exam_name = st.text_input("Nombre del examen (ej. B2 First Complete Test 3)")
    with c2:
        nivel = st.selectbox("Nivel:", ("B1", "B2", "C1", "C2"))
    with c3:
        plantilla = st.selectbox("Plantilla oficial:", (
            "Plantilla 1: Listening Oficial (B1–C2)",
            "Plantilla 2: Reading & Use of English (B1–C2)",
            "Plantilla 3: Test Generator Doble Cara (EX 1–12)"
        ))
    method = st.radio("Origen de las respuestas:", (
        "Subir PDF con respuestas (Cambridge Test Generator / Answer Key)",
        "Pegar respuestas manualmente"
    ))
    extracted = {}
    if method.startswith("Subir PDF"):
        pdf_k = st.file_uploader("Sube el PDF de respuestas", type=("pdf",))
        if pdf_k:
            extracted = parse_pdf_key(pdf_k.read(), plantilla)
    else:
        ph = "Formato: 1: B, 9: WHO, 31: C..." if "Plantilla 3" not in plantilla else "Formato: EX1-1: B, EX1-2: A, EX2-1: WENT..."
        raw = st.text_area("Escribe o pega las soluciones:", placeholder=ph)
        if raw:
            for item in re.split(r'[\n,;]+', raw):
                if ":" in item:
                    k, v = item.split(":", 1)
                    extracted[k.strip().upper()] = v.strip().upper()
    if extracted:
        st.json(extracted)
    if st.button("Guardar Examen", type="primary"):
        if exam_name and extracted:
            st.session_state.exams[exam_name] = {"plantilla": plantilla, "nivel": nivel, "respuestas": extracted}
            save_exams(st.session_state.exams)
            st.success(f"Examen '{exam_name}' guardado correctamente.")
        else:
            st.error("Completa el nombre y las soluciones.")

with tab1:
    sel_exam = st.selectbox("Selecciona el examen:", list(st.session_state.exams.keys()))
    ex_data = st.session_state.exams[sel_exam]
    a_key = ex_data["respuestas"]
    pl_actual = ex_data["plantilla"]
    st.info(f"Nivel: **{ex_data['nivel']}** | Plantilla: **{pl_actual}** | Ítems evaluados: **{len(a_key)}**")

    side_sel = 1
    if "Plantilla 3" in pl_actual:
        side_sel = st.radio("¿Qué cara del Test Generator vas a subir?", (1, 2), format_func=lambda x: f"Cara {x} ({'EX 1–6' if x==1 else 'EX 7–12'})", horizontal=True)

    files = st.file_uploader("Sube PDF de Adobe Scan o fotos de móvil (JPG/PNG):", type=("pdf", "jpg", "jpeg", "png"), accept_multiple_files=True)
    if files:
        imgs = []
        for f in files:
            if f.name.lower().endswith(".pdf"):
                doc = fitz.open(stream=f.read(), filetype="pdf")
                for i, p in enumerate(doc):
                    pix = p.get_pixmap(dpi=200)
                    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    imgs.append((f"{f.name} (Pág {i+1})", cv2.cvtColor(im, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)))
            else:
                arr = np.asarray(bytearray(f.read()), dtype=np.uint8)
                imgs.append((f.name, cv2.imdecode(arr, cv2.IMREAD_COLOR)))

        rows_export = []
        for idx_s, (lbl, im_bgr) in enumerate(imgs):
            if "Plantilla 1" in pl_actual:
                omr, crops, vis, err = process_listening(im_bgr, a_key)
            elif "Plantilla 2" in pl_actual:
                omr, crops, vis, err = process_reading_uoe(im_bgr, a_key)
            else:
                omr, crops, vis, err = process_test_generator(im_bgr, a_key, side=side_sel)

            if err:
                st.error(f"**{lbl}:** {err}")
                continue

            header_im, annot_im = vis
            valid_letters = ("A", "B", "C", "D", "E", "F", "G", "H")
            omr_keys = [k for k, v in a_key.items() if v in valid_letters and (not "Plantilla 3" in pl_actual or (k.startswith(tuple(f"EX{n}-" for n in (range(1,7) if side_sel==1 else range(7,13))))))]
            write_keys = [k for k, v in a_key.items() if v not in valid_letters and k in crops]

            omr_hits = sum(1 for k in omr_keys if omr.get(k) == a_key[k])
            write_hits = 0

            with st.expander(f"Alumno {idx_s+1}: {lbl}", expanded=True):
                c_left, c_right = st.columns((1.2, 1))
                with c_left:
                    st.image(cv2.cvtColor(header_im, cv2.COLOR_BGR2RGB))
                    st.write(f"**Aciertos automáticos (Casillas):** {omr_hits} / {len(omr_keys)}")
                    if write_keys:
                        st.write("**Validación rápida de respuestas escritas:**")
                        for wk in write_keys:
                            ci, cs, cc = st.columns((2, 1.3, 1))
                            with ci:
                                st.image(cv2.cvtColor(crops[wk], cv2.COLOR_BGR2RGB))
                            with cs:
                                st.write(f"**{wk}:** `{a_key[wk]}`")
                            with cc:
                                if st.checkbox("OK", key=f"ok_{idx_s}_{wk}"):
                                    write_hits += 1
                    tot_items = len(omr_keys) + len(write_keys)
                    tot_hits = omr_hits + write_hits
                    nota = round((tot_hits / tot_items) * 10, 2) if tot_items > 0 else 0.0
                    st.success(f"**Nota Total:** {nota}/10 ({tot_hits}/{tot_items} aciertos)")
                    rows_export.append({"Alumno / Archivo": lbl, "Nivel": ex_data["nivel"], "Aciertos": f"{tot_hits}/{tot_items}", "Nota (0-10)": nota})
                with c_right:
                    st.image(cv2.cvtColor(annot_im, cv2.COLOR_BGR2RGB), width=350)

        if rows_export:
            df_res = pd.DataFrame(rows_export)
            st.dataframe(df_res, use_container_width=True)
            buf = io.BytesIO()
            df_res.to_excel(buf, index=False)
            st.download_button("Descargar Notas en Excel (.xlsx)", buf.getvalue(), file_name=f"notas_{ex_data['nivel']}.xlsx")

