# 📚 Proyecto: Biblioteca Virtual Inteligente

Este proyecto es un recomendador de libros híbrido que utiliza **Machine Learning local** para la clasificación y **IA Generativa (Gemini 1.5 Flash)** para las sugerencias.

## 🚀 Cómo probarlo
Puedes acceder a la aplicación desplegada aquí: [PEGA AQUÍ TU ENLACE DE STREAMLIT]

## 🛠️ Tecnologías utilizadas
- **Python / Streamlit** (Interfaz)
- **Scikit-Learn** (Clasificador local Naive Bayes)
- **Google Gemini API** (Generación de recomendaciones)
- **EasyOCR** (Lectura de imágenes)
- **Deepgram API** (Transcripción de voz)

## 📦 Ejecución en local
1. Instalar dependencias: `pip install -r requirements.txt`
2. Entrenar modelo: `python train.py`
3. Lanzar app: `streamlit run app.py`