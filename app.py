import streamlit as st
import joblib
import os
import re
import easyocr
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
import requests

# --- 1. CONFIGURACIÓN DE SEGURIDAD Y RECURSOS ---
load_dotenv()
api_key = os.getenv("API_KEY")
deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")

if not api_key:
    st.error("Error: No se encontró la API_KEY en el archivo .env")
    st.stop()

# Configuración de Gemini 1.5 Flash (Más estable y rápido)
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('gemini-1.5-flash')

@st.cache_resource
def cargar_recursos():
    """Carga el modelo de ML local y el lector OCR una sola vez."""
    try:
        modelo_ml = joblib.load('modelo_libros.pkl')
    except Exception:
        modelo_ml = None
        
    # Inicializamos EasyOCR para español
    lector_ocr = easyocr.Reader(['es'], gpu=False) 
    return modelo_ml, lector_ocr

modelo_local, reader = cargar_recursos()

if modelo_local is None:
    st.warning("⚠️ No se pudo cargar 'modelo_libros.pkl'. Asegúrate de ejecutar train.py primero.")

# --- 2. FUNCIONES DE LÓGICA ---
def es_entrada_valida(texto):
    """Valida que la entrada tenga contenido real."""
    if not texto or len(texto.strip()) < 3:
        return False, "La entrada es demasiado corta para analizar."
    if not re.search(r'[a-zA-ZáéíóúÁÉÍÓÚñÑ]', texto):
        return False, "Por favor, introduce una descripción válida con palabras."
    return True, ""

def procesar_solicitud(texto_entrada):
    """Combina el modelo local para clasificar y Gemini para recomendar."""
    valido, mensaje_error = es_entrada_valida(texto_entrada)
    if not valido:
        return None, mensaje_error
    
    # 1. Predicción con el modelo local mejorado
    categoria = "Literatura General"
    if modelo_local is not None:
        try:
            categoria = modelo_local.predict([texto_entrada])[0]
        except:
            categoria = "Desconocido"

    # 2. Prompt riguroso para Gemini
    prompt = (
        f"Actúa como un experto bibliotecario. El usuario busca libros basados en: '{texto_entrada}'. "
        f"El sistema ha pre-clasificado esto como el género: {categoria}. "
        f"1. Confirma si el género es correcto o ajústalo si es necesario. "
        f"2. Recomienda 3 libros específicos (título y autor) que encajen perfectamente. "
        f"3. Explica en una sola frase breve y atractiva por qué debería leer cada uno. "
        f"Usa un formato limpio con viñetas."
    )
    
    try:
        response = model_gemini.generate_content(prompt)
        if response and response.text:
            return categoria, response.text
        else:
            return categoria, "Lo siento, la IA no pudo generar una respuesta en este momento."
    except Exception as e:
        return categoria, f"Error de conexión con la API de Gemini: {str(e)}"

# --- 3. INTERFAZ DE USUARIO (STREAMLIT) ---
st.set_page_config(page_title="Biblioteca Inteligente", page_icon="📚", layout="centered")

st.title("📚 Mi Biblioteca Virtual Inteligente")
st.markdown("Sistema híbrido de Inteligencia Artificial para recomendaciones literarias.")
st.markdown("---")

tab_txt, tab_img, tab_aud = st.tabs(["✍️ Texto", "📷 Imagen (OCR)", "🎙️ Audio"])

# --- PESTAÑA 1: TEXTO ---
with tab_txt:
    user_input = st.text_area("¿Qué te apetece leer hoy?", 
                              placeholder="Ej: Me encantan las historias de naves espaciales y robots...",
                              height=150)
    if st.button("Analizar y Recomendar", key="btn_texto"):
        with st.spinner("Analizando tu petición con IA..."):
            cat, resultado = procesar_solicitud(user_input)
            if cat:
                st.success(f"🎭 Género detectado: **{cat}**")
                st.markdown(resultado)
            else:
                st.warning(resultado)

# --- PESTAÑA 2: IMAGEN (OCR) ---
with tab_img:
    archivo_img = st.file_uploader("Sube una foto de una contraportada o sinopsis", type=['jpg', 'jpeg', 'png'])
    if archivo_img:
        img_pil = Image.open(archivo_img)
        st.image(img_pil, caption="Imagen cargada", use_container_width=True)
        
        if st.button("Escanear Imagen y Recomendar", key="btn_img"):
            with st.spinner("Extrayendo texto de la imagen..."):
                try:
                    img_array = np.array(img_pil) 
                    resultado_ocr = reader.readtext(img_array, detail=0)
                    texto_extraido = " ".join(resultado_ocr)
                    
                    if texto_extraido.strip():
                        st.info(f"**Texto detectado:** {texto_extraido[:200]}...")
                        cat, resultado = procesar_solicitud(texto_extraido)
                        if cat:
                            st.success(f"🎭 Género detectado: **{cat}**")
                            st.markdown(resultado)
                    else:
                        st.error("No se detectó texto legible en la imagen.")
                except Exception as e:
                    st.error(f"Error en el proceso OCR: {e}")

# --- PESTAÑA 3: AUDIO (DEEPGRAM) ---
with tab_aud:
    archivo_audio = st.file_uploader("Sube un audio describiendo tus gustos", type=['wav', 'mp3', 'm4a'])
    
    if archivo_audio:
        st.audio(archivo_audio)
        if st.button("Transcribir y Analizar", key="btn_aud"):
            if not deepgram_api_key:
                st.error("Falta la API Key de Deepgram.")
            else:
                with st.spinner("Transcribiendo audio..."):
                    try:
                        ext = archivo_audio.name.split('.')[-1].lower()
                        content_type = f"audio/{ext}" if ext != "m4a" else "audio/mp4"

                        headers = {"Authorization": f"Token {deepgram_api_key}", "Content-Type": content_type}
                        params = {"model": "nova-2", "language": "es", "smart_format": "true"}

                        response = requests.post(
                            "https://api.deepgram.com/v1/listen",
                            headers=headers,
                            params=params,
                            data=archivo_audio.read(),
                            timeout=60
                        )

                        if response.status_code == 200:
                            data = response.json()
                            texto_voz = data["results"]["channels"][0]["alternatives"][0]["transcript"]
                            
                            if texto_voz.strip():
                                st.info(f"**Dijiste:** {texto_voz}")
                                cat, resultado = procesar_solicitud(texto_voz)
                                if cat:
                                    st.success(f"🎭 Género detectado: **{cat}**")
                                    st.markdown(resultado)
                            else:
                                st.warning("No se detectó voz en el archivo.")
                        else:
                            st.error(f"Error en Deepgram: {response.status_code}")
                    except Exception as e:
                        st.error(f"Error procesando el audio: {e}")

st.markdown("---")
st.caption("Proyecto Final - Biblioteca Inteligente 2026")