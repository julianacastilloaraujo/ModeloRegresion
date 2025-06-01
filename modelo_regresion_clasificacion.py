# Desarrollado por Juliana Castillo Araujo
# Presentado a Monica Liliana Guzman
# Analítica Aplicada a Negocios

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# === 1. CREACIÓN DE DATOS SIMULADOS ===
np.random.seed(42)
n = 150

data = pd.DataFrame({
    'PIB_per_capita': np.random.normal(15000, 5000, n).clip(1000, 60000),
    'Esperanza_vida': np.random.normal(70, 10, n).clip(40, 90),
    'Acceso_internet': np.random.normal(60, 20, n).clip(0, 100)
})

conditions = [
    (data['PIB_per_capita'] > 25000) & (data['Esperanza_vida'] > 75) & (data['Acceso_internet'] > 75),
    (data['PIB_per_capita'] > 10000) & (data['Esperanza_vida'] > 65),
    (data['PIB_per_capita'] <= 10000)
]
choices = ['Alto', 'Medio', 'Bajo']
data['Nivel_desarrollo'] = np.select(conditions, choices, default='Medio')

# === 2. MODELO DE CLASIFICACIÓN ===
label_encoder = LabelEncoder()
data['Nivel_desarrollo_encoded'] = label_encoder.fit_transform(data['Nivel_desarrollo'])

X = data[['PIB_per_capita', 'Esperanza_vida', 'Acceso_internet']]
y = data['Nivel_desarrollo_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=== REPORTE DEL MODELO ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === 3. VISUALIZACIONES ===
sns.set(style="whitegrid")

def mostrar_grafico(title):
    plt.title(title)
    plt.tight_layout()
    plt.show()

sns.histplot(data['PIB_per_capita'], kde=True)
mostrar_grafico("Distribución del PIB per cápita")

sns.histplot(data['Esperanza_vida'], kde=True)
mostrar_grafico("Distribución de la Esperanza de Vida")

sns.histplot(data['Acceso_internet'], kde=True)
mostrar_grafico("Distribución del Acceso a Internet")

sns.boxplot(x='Nivel_desarrollo', y='PIB_per_capita', data=data)
mostrar_grafico("PIB per cápita por Nivel de Desarrollo")

sns.boxplot(x='Nivel_desarrollo', y='Esperanza_vida', data=data)
mostrar_grafico("Esperanza de Vida por Nivel de Desarrollo")

sns.boxplot(x='Nivel_desarrollo', y='Acceso_internet', data=data)
mostrar_grafico("Acceso a Internet por Nivel de Desarrollo")

sns.scatterplot(data=data, x='PIB_per_capita', y='Esperanza_vida', hue='Nivel_desarrollo')
mostrar_grafico("PIB vs Esperanza de Vida")

sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
mostrar_grafico("Matriz de Correlación")

sns.countplot(x='Nivel_desarrollo', data=data)
mostrar_grafico("Distribución del Nivel de Desarrollo")

conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
mostrar_grafico("Matriz de Confusión del Modelo")
