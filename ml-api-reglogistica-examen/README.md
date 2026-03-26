# ML API Regresión Logística Examen

Proyecto de pipeline de machine learning para predicción de aprobación en exámenes basado en horas de estudio usando regresión logística.

## Estructura

- `data/`: Datos de entrada
- `models/`: Modelos entrenados
- `src/`: Código fuente modular
- `api/`: API de FastAPI para servir predicciones
- `notebooks/`: Análisis exploratorio

## Uso

1. Entrenar el modelo: Ejecutar `src/pipeline/training_pipeline.py`
2. Ejecutar la API: `uvicorn api.main:app --reload`
3. Para la app Streamlit: Ejecutar `streamlit run app.py` (actualizado)