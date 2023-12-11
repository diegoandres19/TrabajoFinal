import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

nltk.download('punkt')
nltk.download('stopwords')

data = [
    ("Oferta especial, ¡compra ahora y obtén un descuento del 50%!", "spam"),
    ("Reunión de equipo a las 3 pm en la sala de conferencias", "no_spam"),
    ("Gana un iPhone gratis, solo hoy", "spam"),
    ("Recordatorio: Pago de la factura de electricidad mañana", "no_spam"),
    ("Hola, Carlos. ¿Cómo estás?", "no_spam"),
    ("caac", "no_spam"),
    ("¡Felicidades! Has ganado un millón de dólares", "spam"),
    ("Venta exclusiva: solo por hoy, descuentos del 70%", "spam"),
    ("Actualización importante: nueva versión del software disponible", "no_spam"),
    ("¡Tu préstamo ha sido aprobado! Obtén efectivo rápido", "spam"),
    ("Recordatorio de reunión: mañana a las 10 am", "no_spam"),
    ("¡Gana un viaje todo incluido! Participa ahora", "spam"),
    ("Promoción especial para clientes leales", "spam"),
    ("Informe mensual de ventas disponible para revisión", "no_spam"),
    ("Descarga gratuita: software antivirus premium", "spam"),
    ("¡Increíble! Gana dinero rápido y seguro", "spam"),
    ("Oferta exclusiva: obtén ingresos pasivos ahora mismo", "spam"),
    ("Préstamos rápidos y fáciles, ¡sin riesgo!", "spam"),
    ("¡Actúa ahora! No pierdas la oportunidad de aumentar tus ingresos", "spam"),
    ("¡No te lo pierdas! Haz clic aquí para obtener beneficios garantizados", "spam"),
    ("Ganador garantizado: ¡sé el ganador y obtén tu premio ahora!", "spam"),
    ("Dinero fácil y sin complicaciones. ¡Hazlo tú mismo!", "spam"),
    ("¡No más deudas! Elimina tus deudas fácilmente", "spam"),
    ("Viagra y Cialis: solución milagrosa para aumentar la potencia", "spam"),
    ("Compra directa al fabricante: ¡ahorra con esta oferta exclusiva!", "spam"),
    ("Gana en grande con esta increíble oferta. ¡Hazlo ahora mismo!", "spam"),
    ("Este es un mensaje normal sin palabras clave de spam", "no_spam"),
    ("Hola, ¿cómo estás? Espero verte pronto", "no_spam"),
    ("Recordatorio de reunión mañana a las 2 pm", "no_spam"),
    ("¡Hazte rico rápido! Participa en esta oportunidad única", "spam"),
    ("¡Increíble pero cierto! Gana premios exclusivos", "spam"),
    ("Compra directa al fabricante: productos aprobados", "spam"),
    ("¡Gana dinero sin esfuerzo! Ingresos pasivos garantizados", "spam"),
    ("Prueba gratis el nuevo producto revolucionario", "spam"),
    ("Oferta especial: solo por hoy, descuentos del 80%", "spam"),
    ("¡Hazlo tú mismo y ahorra dinero! Oferta única", "spam"),
    ("¡La verdad revelada! Descubre el secreto del éxito", "spam"),
    ("Gana seguidores ahora mismo. ¡Hazlo fácilmente!", "spam"),
    ("Elimina arrugas con esta increíble solución", "spam"),
    ("¡Hazlo instantáneo! Resultados garantizados", "spam"),
    ("Hola, ¿cómo estás? Espero verte pronto", "no_spam"),
    ("Recordatorio de reunión mañana a las 2 pm", "no_spam"),
    ("Este es un mensaje normal sin palabras clave de spam", "no_spam"),
    ("Descarga gratuita: aplicación útil para tu dispositivo", "no_spam"),
    ("Promoción especial para nuestros clientes leales", "no_spam"),
    ("Informe mensual de proyectos disponible para revisión", "no_spam"),
]


#PROCESAR TEXTO
def preprocess_text(text):
    stop_words = set(stopwords.words('spanish'))
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

texts, labels = zip(*data)
texts = [preprocess_text(text) for text in texts]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy:.2f}')

print('\nClassification Report:')
print(classification_report(y_test, predictions))

user_input = input("Ingresa un texto: ")
user_input = preprocess_text(user_input)
user_vector = vectorizer.transform([user_input])
prediction = classifier.predict(user_vector)[0]

print(f'\nResultado: "{user_input}" es {prediction}')