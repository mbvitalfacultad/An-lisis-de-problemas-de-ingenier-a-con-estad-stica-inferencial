# simulador.py
# Simulador avanzado de IC (carga Excel)
# Nombre provisional: "Para resolver problemas de ingeniería"

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from fpdf import FPDF
import io

st.set_page_config(page_title="Para resolver problemas de ingeniería", layout="wide")

st.title("Simulador de Intervalos de Confianza para Ingeniería")
st.markdown("""
Este simulador permite:
- Cargar un archivo Excel con datos de una variable.
- Analizar la distribución (histograma y test de normalidad Anderson-Darling).
- Construir distintos intervalos de confianza:
  - Media (varianza conocida)
  - Media (varianza desconocida)
  - Varianza
  - Desvío estándar
  - Proporción poblacional
- Elegir el nivel de confianza.
- Visualizar resultados numéricos y gráficos.
- Descargar resultados en Excel o PDF.
""")

# --- Carga de archivo Excel ---
uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx", "xls"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("Archivo cargado exitosamente.")
    st.write("Vista previa de los datos:")
    st.dataframe(df.head())
    
    column = st.selectbox("Seleccionar columna de datos:", df.columns)
    datos = df[column].dropna().values
    
    st.subheader("Análisis de distribución")
    # Histograma
    fig, ax = plt.subplots()
    ax.hist(datos, bins='auto', color='skyblue', edgecolor='black')
    ax.axvline(np.mean(datos), color='red', linestyle='dashed', linewidth=2, label="Media muestral")
    ax.set_title(f"Histograma de {column}")
    ax.legend()
    st.pyplot(fig)
    
    # Test de normalidad
    ad_result = stats.anderson(datos)
    st.write("Test de Anderson-Darling:")
    st.write(f"Estadístico: {ad_result.statistic:.4f}")
    for sl, cv in zip(ad_result.significance_level, ad_result.critical_values):
        st.write(f"Nivel de significancia: {sl}%, Valor crítico: {cv}")
    
    # --- Selección de tipo de IC ---
    st.subheader("Construcción de Intervalos de Confianza")
    tipo_ic = st.selectbox("Seleccionar tipo de IC:",
                           ["Media (varianza conocida)", 
                            "Media (varianza desconocida)",
                            "Varianza",
                            "Desvío estándar",
                            "Proporción poblacional"])
    nivel_conf = st.slider("Nivel de confianza (%)", 50, 99, 95)
    alpha = 1 - nivel_conf/100
    
    n = len(datos)
    x_bar = np.mean(datos)
    s = np.std(datos, ddof=1)
    
    ic_inferior = None
    ic_superior = None
    
    if tipo_ic == "Media (varianza conocida)":
        sigma = s  # Para ejemplo, se usa s como sigma conocido
        z = stats.norm.ppf(1-alpha/2)
        ic_inferior = x_bar - z*sigma/np.sqrt(n)
        ic_superior = x_bar + z*sigma/np.sqrt(n)
        
    elif tipo_ic == "Media (varianza desconocida)":
        t = stats.t.ppf(1-alpha/2, df=n-1)
        ic_inferior = x_bar - t*s/np.sqrt(n)
        ic_superior = x_bar + t*s/np.sqrt(n)
        
    elif tipo_ic == "Varianza":
        chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
        chi2_upper = stats.chi2.ppf(1-alpha/2, df=n-1)
        ic_inferior = (n-1)*s**2 / chi2_upper
        ic_superior = (n-1)*s**2 / chi2_lower
        
    elif tipo_ic == "Desvío estándar":
        chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
        chi2_upper = stats.chi2.ppf(1-alpha/2, df=n-1)
        ic_inferior = np.sqrt((n-1)*s**2 / chi2_upper)
        ic_superior = np.sqrt((n-1)*s**2 / chi2_lower)
        
    elif tipo_ic == "Proporción poblacional":
        p_hat = x_bar  # Asumiendo 0/1 datos para proporción
        z = stats.norm.ppf(1-alpha/2)
        ic_inferior = p_hat - z*np.sqrt(p_hat*(1-p_hat)/n)
        ic_superior = p_hat + z*np.sqrt(p_hat*(1-p_hat)/n)
    
    st.write(f"Intervalo de confianza ({nivel_conf}%): [{ic_inferior:.4f}, {ic_superior:.4f}]")
    
    # Gráfico IC
    fig2, ax2 = plt.subplots()
    ax2.hist(datos, bins='auto', color='lightgreen', edgecolor='black')
    ax2.axvline(x_bar, color='red', linestyle='dashed', linewidth=2, label="Media muestral")
    ax2.axvline(ic_inferior, color='blue', linestyle='dotted', linewidth=2, label="IC inferior")
    ax2.axvline(ic_superior, color='blue', linestyle='dotted', linewidth=2, label="IC superior")
    ax2.set_title(f"IC de {tipo_ic} para {column}")
    ax2.legend()
    st.pyplot(fig2)
    
    # --- Descarga de resultados ---
    st.subheader("Descargar resultados")
    
    # Excel
    if st.button("Generar Excel con resultados"):
        output = io.BytesIO()
        df_out = df.copy()
        df_out["Media_muestral"] = x_bar
        df_out["Desviacion_muestral"] = s
        df_out["IC_inferior"] = ic_inferior
        df_out["IC_superior"] = ic_superior
        df_out.to_excel(output, index=False)
        output.seek(0)
        st.download_button("Descargar Excel", data=output, file_name="resultados_IC.xlsx")
    
    # PDF
    if st.button("Generar PDF con resultados"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Resultados de IC para {column}", ln=True)
        pdf.cell(0, 10, f"Media muestral: {x_bar:.4f}", ln=True)
        pdf.cell(0, 10, f"Desviación muestral: {s:.4f}", ln=True)
        pdf.cell(0, 10, f"IC ({nivel_conf}%): [{ic_inferior:.4f}, {ic_superior:.4f}]", ln=True)
        # Guardar gráfico
        fig2.savefig("grafico_IC.png")
        pdf.image("grafico_IC.png", x=10, y=60, w=180)
        pdf.output("resultados_IC.pdf")
        st.download_button("Descargar PDF", data=open("resultados_IC.pdf","rb"), file_name="resultados_IC.pdf")
