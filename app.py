import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Regresión Lineal", layout="centered")

st.title("📊 Predicción de Notas según Horas de Estudio")

# Cargar datos
df = pd.read_csv("data/horas_estudio.csv")

st.subheader("📁 Datos")
st.write(df)

# Separar variables
X = df[['Horas']]
y = df['Notas']

# Modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Predicción
st.subheader("🔮 Predicción")

horas = st.slider("Horas de estudio", 0, 10, 5)
prediccion = modelo.predict([[horas]])

st.write(f"📌 Nota estimada: **{prediccion[0]:.2f}**")

# Gráfica
st.subheader("📈 Gráfica de Regresión")

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, modelo.predict(X))
ax.set_xlabel("Horas de estudio")
ax.set_ylabel("Notas")

st.pyplot(fig)