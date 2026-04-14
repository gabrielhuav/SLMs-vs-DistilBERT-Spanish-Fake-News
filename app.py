import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import base64
from io import BytesIO

# --- CONFIGURACIÓN E INFRAESTRUCTURA ---
OLLAMA_URL = "http://ollama:11434/api/generate"
MODELO_LLM = "llama3.2"

# Sesión HTTP persistente para evitar saturación de puertos en 61k registros
http_session = requests.Session()

st.set_page_config(page_title="Evaluador Comparativo LLM", layout="wide")

# --- FUNCIONES DE SOPORTE PARA REPORTES ---
def fig_to_base64(fig):
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def generate_academic_report(df_detailed, df_macro, acc, cm_base64, model, total_time):
    """Genera un reporte HTML profesional con métricas detalladas y promedios macro"""
    html = f"""
    <html>
    <head>
        <title>Academic Evaluation Report - {model}</title>
        <style>
            body {{ font-family: 'Times New Roman', serif; margin: 40px; line-height: 1.6; color: #333; }}
            .container {{ max-width: 950px; margin: auto; background: white; padding: 40px; border: 1px solid #ddd; }}
            h1, h2, h3 {{ text-align: center; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .stats {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-left: 10px solid #3498db; font-size: 1.2em; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; margin-bottom: 30px; }}
            th, td {{ border: 1px solid #000; padding: 10px; text-align: center; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .figure {{ text-align: center; margin-top: 40px; font-style: italic; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Academic Performance Report</h1>
            <p><strong>Model:</strong> {model}</p>
            <p><strong>Execution Time:</strong> {total_time:.2f} minutes</p>
            
            <div class="stats">
                <strong>Overall Accuracy:</strong> {acc:.4f}
            </div>

            <h3>Table 1. Detailed Performance Metrics per Class</h3>
            {df_detailed.to_html(index=False)}

            <h3>Table 2. Summary Metrics (Macro Average)</h3>
            {df_macro.to_html(index=False)}

            <div class="figure">
                <img src="data:image/png;base64,{cm_base64}" width="100%">
                <p>Figure 1. Confusion Matrix for Multiclass Classification.</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

# --- LÓGICA DE INFERENCIA ---
def clasificar_texto_llm(texto):
    prompt = f"""Eres un clasificador experto de noticias en español. 
Clasifica el siguiente texto en una de estas tres categorías:
0 = Falso (Fake News)
1 = Verdadero (Noticia Real)
2 = Sátira (Parodia)

Reglas estrictas: Responde ÚNICAMENTE con el número 0, 1 o 2. No agregues texto ni explicaciones.

Texto: {texto}"""
    
    payload = {
        "model": MODELO_LLM, 
        "prompt": prompt, 
        "stream": False, 
        "options": {"temperature": 0.0}
    }
    
    try:
        response = http_session.post(OLLAMA_URL, json=payload, timeout=15)
        if response.status_code == 200:
            res = response.json()["response"].strip()
            for d in ["0", "1", "2"]:
                if d in res: return int(d)
        return -1
    except: return -1

# --- INTERFAZ ---
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "mensajes_chat" not in st.session_state:
    st.session_state.mensajes_chat = []

st.title("Framework de Evaluación Comparativa: DistilBERT vs LLM Local")
st.write(f"Motor: **{MODELO_LLM.upper()}** | Pipeline Optimizado para 61k registros")

tab1, tab2, tab3 = st.tabs(["Evaluación Individual", "Evaluación por Lotes", "💬 Chat"])

# --- PESTAÑA 2: EVALUACIÓN MASIVA ---
with tab2:
    archivo_csv = st.file_uploader("Subir Test Set (.csv)", type=['csv'])
    
    if archivo_csv:
        df_test = pd.read_csv(archivo_csv, sep=';').dropna(subset=['text', 'label'])
        num = st.slider("Registros a evaluar:", 10, len(df_test), len(df_test))
        
        if st.button("🚀 Iniciar Evaluación Masiva", type="primary"):
            df_eval = df_test.head(num).copy()
            predicciones = []
            bar = st.progress(0)
            status = st.empty()
            start = time.time()
            
            for idx, (i, row) in enumerate(df_eval.iterrows()):
                predicciones.append(clasificar_texto_llm(str(row['text'])))
                
                if (idx + 1) % 100 == 0 or (idx + 1) == num:
                    bar.progress((idx + 1) / num)
                    status.write(f"Procesando: {idx + 1} / {num}")
            
            df_eval['prediccion_llm'] = predicciones
            df_limpio = df_eval[df_eval['prediccion_llm'] != -1]
            st.session_state.batch_results = {
                'df': df_limpio, 'time': (time.time() - start)/60, 'model': MODELO_LLM
            }

    if st.session_state.batch_results:
        res = st.session_state.batch_results
        y_true, y_pred = res['df']['label'].astype(int), res['df']['prediccion_llm'].astype(int)
        
        st.divider()
        idioma = st.radio("Selecciona idioma de etiquetas:", ["Español", "English (Academic)"], horizontal=True)
        
        labels_raw = ['FALSO (0)', 'REAL (1)', 'SÁTIRA (2)']
        labels_eng = ['FAKE (0)', 'REAL (1)', 'SATIRE (2)']
        current_labels = labels_raw if idioma == "Español" else labels_eng
        
        # --- CÁLCULOS ---
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2], zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        
        # Specificity por clase
        spec_list = []
        for i in range(3):
            tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
            fp = np.sum(cm[:, i]) - cm[i, i]
            spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        # TABLA 1: DETALLADA
        df_detailed = pd.DataFrame({
            'Class': current_labels,
            'Precision': p,
            'Recall (Sensitivity)': r,
            'F1-Score': f1,
            'Specificity': spec_list
        })

        # TABLA 2: MACRO
        df_macro = pd.DataFrame({
            'Metric Type': ['Macro Average'],
            'Precision': [np.mean(p)],
            'Recall (Sensitivity)': [np.mean(r)],
            'F1-Score': [np.mean(f1)],
            'Specificity': [np.mean(spec_list)]
        })
        
        # --- VISUALIZACIÓN ---
        st.subheader("Performance Results")
        st.info(f"**Overall Accuracy:** {acc:.4f}")
        
        st.write("**1. Detailed Metrics by Class:**")
        st.dataframe(df_detailed.style.format({c: '{:.4f}' for c in df_detailed.columns if c != 'Class'}), use_container_width=True)
        
        st.write("**2. Macro Average Metrics:**")
        st.dataframe(df_macro.style.format({c: '{:.4f}' for c in df_macro.columns if c != 'Metric Type'}), use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=current_labels, yticklabels=current_labels, ax=ax)
        ax.set_title(f"Confusion Matrix - {res['model'].upper()}")
        st.pyplot(fig)

        # --- EXPORTACIÓN ---
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 CSV Results", data=res['df'].to_csv(index=False, sep=';').encode('utf-8'), file_name="full_results.csv")
        with c2:
            cm_b64 = fig_to_base64(fig)
            # El reporte académico siempre se genera con los nombres en inglés
            df_det_eng = df_detailed.copy()
            df_det_eng['Class'] = labels_eng
            report_html = generate_academic_report(df_det_eng, df_macro, acc, cm_b64, res['model'], res['time'])
            st.download_button("📄 ACADEMIC REPORT (HTML)", data=report_html, file_name="Complete_Academic_Report.html", mime="text/html")

# --- PESTAÑAS 1 Y 3 ---
with tab1:
    txt = st.text_area("Cuerpo de la noticia:", height=150)
    if st.button("Analizar Individualmente"):
        res_ind = clasificar_texto_llm(txt)
        if res_ind == 0: st.error("0 - FALSO")
        elif res_ind == 1: st.success("1 - REAL")
        elif res_ind == 2: st.warning("2 - SÁTIRA")

with tab3:
    cont = st.container(height=400, border=True)
    for m in st.session_state.mensajes_chat:
        with cont: st.chat_message(m["role"]).markdown(m["content"])
    if p_chat := st.chat_input("Chat con LLM..."):
        st.session_state.mensajes_chat.append({"role": "user", "content": p_chat})
        with cont: st.chat_message("user").markdown(p_chat)
        with cont:
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_r = ""
                r_chat = requests.post(OLLAMA_URL, json={"model": MODELO_LLM, "prompt": p_chat, "stream": True}, stream=True)
                for line in r_chat.iter_lines():
                    if line:
                        full_r += json.loads(line.decode())["response"]
                        placeholder.markdown(full_r + "▌")
                placeholder.markdown(full_r)
                st.session_state.mensajes_chat.append({"role": "assistant", "content": full_r})