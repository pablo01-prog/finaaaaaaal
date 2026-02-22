import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# 1. Dataset AMPLIADO y más específico
data = {
    'texto' : [
        # Fantasía
        'magia dragones espada guerrero aventura hechizo varita elfo enano mundo magico orcos gnomos cronicas reino',
        'una historia de magos y dragones con espadas legendarias y mucha aventura caballeros fantasia',
        # Policial / Detectives
        'crimen detective asesinato misterio policia huellas culpable investigacion forense thriller suspense noir inspector pistas',
        'un detective busca al asesino en un misterio policial lleno de intriga resolucion de casos',
        # Romance
        'amor romance pareja enamorados boda pasion corazon novios cita romantica sentimientos drama amoroso besos',
        'historia de amor sobre una pareja de enamorados que planean su boda enamoramiento jovenes adultos',
        # Ciencia Ficción
        'futuro naves espaciales robots planetas galaxia tecnologia alienigenas cosmos distopia ciberpunk interestelar marte',
        'viaje al futuro en naves espaciales con robots inteligentes y otros planetas colonizacion espacial',
        # Terror
        'fantasmas terror miedo susto sangre oscuro pesadilla monstruo espiritu grito posesion casa maldita paranormal',
        'un relato de terror con fantasmas y monstruos en un ambiente oscuro y de miedo horror psicologico',
        # Histórica
        'historia antigua guerra reyes imperio epoca medieval caballero batalla siglo biografia revolucion pasado archivos',
        'narración sobre la historia antigua con reyes y batallas de un imperio caido hechos reales victoria'
    ],
    'genero': [
        'Fantasia', 'Fantasia', 'Policial', 'Policial', 
        'Romance', 'Romance', 'Ciencia Ficcion', 'Ciencia Ficcion',
        'Terror', 'Terror', 'Historica', 'Historica'
    ]
}

df = pd.DataFrame(data)

# Mejoramos el pipeline para ignorar stop_words en español
modelo = make_pipeline(
    TfidfVectorizer(
        lowercase=True, 
        strip_accents="unicode", 
        ngram_range=(1, 2),
        stop_words=['de', 'la', 'el', 'un', 'una', 'y', 'en', 'con']
    ), 
    MultinomialNB(alpha=0.1) # Alpha bajo para que sea más sensible
)

print("Entrenando el modelo de predicción mejorado...")
modelo.fit(df['texto'], df['genero'])

joblib.dump(modelo, 'modelo_libros.pkl')
print("✅ Nuevo 'modelo_libros.pkl' generado.")