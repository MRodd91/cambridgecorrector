import streamlit as st
import cv2
import numpy as np
import fitz
import pandas as pd
import json
import os
import re
import io

st.set_page_config(page_title="Corrector Cambridge A2–C1", layout="wide")

DB_FILE = "banco_examenes.json"

def load_exams():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "Demo A2 Key Reading (10 preguntas - Modelo A)": {
            "modelo": "Modelo A (Tipo Test A-H)",
            "nivel": "A2",
            "respuestas": {
                "1": "B", "2": "A", "3": "C", "4": "B", "5": "A",
                "6": "C", "7": "B", "8": "A", "9": "C", "10": "B"
            }
        },
        "Demo B2 Use of English Part 2 (8 huecos - Modelo B)": {
            "modelo": "Modelo B (Escritura 1-30)",
            "nivel": "B2",
            "respuestas": {
                "1": "WHO", "2": "WHICH", "3": "DESPITE", "4": "BEEN",
                "5": "SUCH", "6": "ALTHOUGH", "7": "MAKE", "8": "TAKEN"
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
    min_area = (w_img * h_img) * 0.00025
    max_area = (w_img * h_img) * 0.006

    fiducials = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.72 <= aspect_ratio <= 1.35:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        fiducials.append((cx, cy))

    if len(fiducials) < 4:
        return None, "No se detectaron las 4 marcas negras de las esquinas. Asegúrate de que salgan enteras en la foto."

    pts = np.array(fiducials, dtype="float32")
    corners = order_points(pts)
    warped = four_point_transform(img_bgr, corners)
    return warped, None

def grade_model_a(img_bgr, answer_key):
    warped, error = align_sheet(img_bgr)
    if error:
        return None, None, error

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_thresh = cv2.adaptiveThreshold(
        warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    header_crop = warped[110:255, 30:1170]
    letters = ("A", "B", "C", "D", "E", "F", "G", "H")
    col_x_starts = (118, 495, 872)
    col_x_step = 35.5
    row_y_start = 428
    row_y_step = 53.2

    student_answers = {}
    annotated = warped.copy()

    for q_num in range(1, 61):
        col_idx = (q_num - 1) // 20
        row_idx = (q_num - 1) % 20
        cy = int(row_y_start + row_idx * row_y_step)

        fill_ratios = []
        boxes_coords = []
        for opt_idx in range(8):
            cx = int(col_x_starts[col_idx] + opt_idx * col_x_step)
            x1, y1, x2, y2 = cx - 12, cy - 9, cx + 12, cy + 9
            roi = warped_thresh[y1:y2, x1:x2]
            fill_ratio = cv2.countNonZero(roi) / float(roi.size)
            fill_ratios.append(fill_ratio)
            boxes_coords.append((x1, y1, x2, y2))

        max_fill = max(fill_ratios)
        chosen_idx = int(np.argmax(fill_ratios))
        chosen_letter = letters[chosen_idx] if max_fill > 0.40 else "-"
        student_answers[str(q_num)] = chosen_letter

        if str(q_num) in answer_key:
            correct_letter = answer_key[str(q_num)].upper()
            if correct_letter in letters:
                correct_idx = letters.index(correct_letter)
                cx1, cy1, cx2, cy2 = boxes_coords[correct_idx]
                if chosen_letter == correct_letter:
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 180, 0), 3)
                else:
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 180, 0), 2)
                    if chosen_letter != "-":
                        wx1, wy1, wx2, wy2 = boxes_coords[chosen_idx]
                        cv2.rectangle(annotated, (wx1, wy1), (wx2, wy2), (0, 0, 220), 3)

    return student_answers, (header_crop, annotated), None

def crop_model_b(img_bgr, max_q):
    warped, error = align_sheet(img_bgr)
    if error:
        return None, None, error

    header_crop = warped[110:255, 30:1170]
    crops = {}
    col_x_starts = (145, 725)
    box_w = 430
    box_h = 62
    row_y_start = 415
    row_y_step = 82.5

    for q_num in range(1, min(max_q, 30) + 1):
        col_idx = (q_num - 1) // 15
        row_idx = (q_num - 1) % 15
        x1 = col_x_starts[col_idx]
        y1 = int(row_y_start + row_idx * row_y_step)
        crops[str(q_num)] = warped[y1:y1 + box_h, x1:x1 + box_w]

    return crops, (header_crop, warped), None

def parse_cambridge_pdf_key(pdf_bytes, is_model_a=True):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = "\n".join([page.get_text() for page in doc])
    key = {}
    if is_model_a:
        matches = re.findall(r'\b([1-5]?[0-9]|60)\s*[\.\)\:\-]?\s*([A-H])\b', full_text)
        for num_str, letter in matches:
            if num_str not in key:
                key[num_str] = letter.upper()
    else:
        lines = full_text.splitlines()
        for line in lines:
            m = re.match(r'^\s*([1-2]?[0-9]|30)\s*[\.\)\:\-]\s*(.+)$', line.strip())
            if m and m.group(1) not in key:
                key[m.group(1)] = m.group(2).strip().upper()
    return key
st.title("Corrector Cambridge A2–C1 — Academia")
st.caption("Compatible con Modelo A (Tipo Test A–H), Modelo B (Escritura), fotos de móvil y Adobe Scan.")

if "exams" not in st.session_state:
    st.session_state.exams = load_exams()

tab1, tab2 = st.tabs(("1. Corregir Exámenes", "2. Cargar Examen / Test Generator"))

with tab2:
    st.subheader("Configurar respuestas de un examen nuevo")
    c_meta1, c_meta2, c_meta3 = st.columns((2, 1, 1.5))
    with c_meta1:
        exam_name = st.text_input("Nombre del examen (ej. B1 Preliminary Compact Test 2)")
    with c_meta2:
        nivel = st.selectbox("Nivel:", ("A2", "B1", "B2", "C1"))
    with c_meta3:
        modelo = st.selectbox("Plantilla a usar:", ("Modelo A (Tipo Test A-H)", "Modelo B (Escritura 1-30)"))

    is_a = "Modelo A" in modelo
    method = st.radio("¿Cómo cargar las soluciones?", (
        "Subir PDF con respuestas del Cambridge Test Generator",
        "Escribir o pegar respuestas manualmente"
    ))

    extracted_key = {}
    if method == "Subir PDF con respuestas del Cambridge Test Generator":
        key_pdf = st.file_uploader("Sube el PDF del Answer Key de Cambridge", type=("pdf",))
        if key_pdf:
            extracted_key = parse_cambridge_pdf_key(key_pdf.read(), is_model_a=is_a)
            if extracted_key:
                st.success(f"Detectadas {len(extracted_key)} respuestas automáticamente.")
            else:
                st.warning("No se detectaron respuestas automáticamente. Puedes escribirlas abajo.")
    else:
        placeholder = "Ejemplo Modelo A: B, A, D, C, A... o 1:B, 2:A" if is_a else "Ejemplo Modelo B (una por línea): 1. WHO"
        raw_text = st.text_area("Introduce las respuestas:", placeholder=placeholder)
        if raw_text:
            if is_a:
                tokens = re.findall(r'[A-H]', raw_text.upper())
                extracted_key = {str(i + 1): val for i, val in enumerate(tokens[:60])}
            else:
                lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
                for i, l in enumerate(lines[:30]):
                    clean = re.sub(r'^\d+[\.\)\:\-\s]+', '', l)
                    extracted_key[str(i + 1)] = clean.upper()

    if extracted_key:
        st.write("**Vista previa de soluciones cargadas:**")
        st.json(extracted_key)

    if st.button("Guardar Examen en el Banco Compartido", type="primary"):
        if exam_name and extracted_key:
            st.session_state.exams[exam_name] = {
                "modelo": modelo,
                "nivel": nivel,
                "respuestas": extracted_key
            }
            save_exams(st.session_state.exams)
            st.success(f"Examen '{exam_name}' ({nivel}) guardado correctamente.")
        else:
            st.error("Escribe un nombre para el examen y carga al menos una respuesta.")

with tab1:
    selected_exam = st.selectbox("Selecciona el examen a corregir:", list(st.session_state.exams.keys()))
    exam_data = st.session_state.exams[selected_exam]
    answer_key = exam_data["respuestas"]
    modelo_actual = exam_data["modelo"]
    st.info(f"Nivel: **{exam_data['nivel']}** | Plantilla: **{modelo_actual}** | Total preguntas: **{len(answer_key)}**")

    uploaded_files = st.file_uploader(
        "Sube el PDF de Adobe Scan o selecciona las fotos de los alumnos (JPG/PNG):",
        type=("pdf", "jpg", "jpeg", "png"),
        accept_multiple_files=True
    )

    if uploaded_files:
        images_to_process = []
        for f in uploaded_files:
            if f.name.lower().endswith(".pdf"):
                doc = fitz.open(stream=f.read(), filetype="pdf")
                for i, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=200)
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
                    images_to_process.append((f"{f.name} - Pág {i+1}", img))
            else:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                images_to_process.append((f.name, img))

        if "Modelo A" in modelo_actual:
            if st.button("Corregir Lote Automáticamente", type="primary"):
                results_table = []
                for label, img_bgr in images_to_process:
                    answers, visuals, error = grade_model_a(img_bgr, answer_key)
                    if error:
                        st.error(f"**{label}:** {error}")
                        continue

                    header_crop, annotated = visuals
                    hits = sum(1 for q, corr in answer_key.items() if answers.get(q) == corr)
                    total = len(answer_key)
                    score_10 = round((hits / total) * 10, 2)
                    wrong_qs = [f"P{q} ({answers.get(q)}≠{corr})" for q, corr in answer_key.items() if answers.get(q) != corr]

                    results_table.append({
                        "Alumno / Archivo": label,
                        "Nivel": exam_data["nivel"],
                        "Aciertos": f"{hits}/{total}",
                        "Nota (0-10)": score_10,
                        "Fallos": ", ".join(wrong_qs) if wrong_qs else "Ninguno"
                    })

                    with st.expander(f"{label} — Nota: {score_10}/10 ({hits}/{total})", expanded=True):
                        c1, c2 = st.columns((1, 1.3))
                        with c1:
                            st.write("**Cabecera del alumno:**")
                            st.image(cv2.cvtColor(header_crop, cv2.COLOR_BGR2RGB))
                            st.write("**Errores:**", ", ".join(wrong_qs) if wrong_qs else "¡Todo correcto!")
                        with c2:
                            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), width=350)

                if results_table:
                    df = pd.DataFrame(results_table)
                    st.subheader("Resultados del Grupo")
                    st.dataframe(df, use_container_width=True)
                    excel_buf = io.BytesIO()
                    df.to_excel(excel_buf, index=False)
                    st.download_button(
                        "Descargar Excel con Notas (.xlsx)",
                        excel_buf.getvalue(),
                        file_name=f"notas_{exam_data['nivel']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.subheader("Validación Rápida de Escritura (Modelo B)")
            max_q = max(int(k) for k in answer_key.keys())
            results_b = []
            for idx_st, (label, img_bgr) in enumerate(images_to_process):
                crops, visuals, error = crop_model_b(img_bgr, max_q)
                if error:
                    st.error(f"**{label}:** {error}")
                    continue
                header_crop, _ = visuals
                with st.expander(f"Alumno {idx_st + 1}: {label}", expanded=True):
                    st.image(cv2.cvtColor(header_crop, cv2.COLOR_BGR2RGB), width=550)
                    hits = 0
                    for q_num in range(1, max_q + 1):
                        q_str = str(q_num)
                        if q_str in answer_key and q_str in crops:
                            col_img, col_sol, col_chk = st.columns((2, 1.5, 1))
                            with col_img:
                                st.image(cv2.cvtColor(crops[q_str], cv2.COLOR_BGR2RGB))
                            with col_sol:
                                st.write(f"**P{q_str} Solución:** `{answer_key[q_str]}`")
                            with col_chk:
                                ok = st.checkbox("Correcta", key=f"chk_{idx_st}_{q_str}")
                                if ok:
                                    hits += 1
                    score_10 = round((hits / len(answer_key)) * 10, 2)
                    st.write(f"**Nota actual:** {score_10}/10 ({hits}/{len(answer_key)})")
                    results_b.append({
                        "Alumno / Archivo": label,
                        "Nivel": exam_data["nivel"],
                        "Aciertos": f"{hits}/{len(answer_key)}",
                        "Nota (0-10)": score_10
                    })
            if results_b:
                df_b = pd.DataFrame(results_b)
                excel_buf = io.BytesIO()
                df_b.to_excel(excel_buf, index=False)
                st.download_button(
                    "Descargar Excel con Notas (.xlsx)",
                    excel_buf.getvalue(),
                    file_name=f"notas_escritura_{exam_data['nivel']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
