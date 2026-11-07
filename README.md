# Sentinel Blue — Análisis de sentimiento en tweets

Proyecto final en equipo (3 personas) sobre el dataset **Sentiment140** (`training.1600000.processed.noemoticon.csv`): 1.6 millones de tweets etiquetados como negativo (`target = 0`) o positivo (`target = 4`).

Paleta: azul oscuro (`#06111f`, `#0d2744`), azul claro (`#3db8ff`, `#9ad8ff`) y acentos fosforescentes (`#39ffb0`, `#00f0ff`).

---

## Cómo está dividido el trabajo

| Parte | Quién | Entregable |
| --- | --- | --- |
| **1** | Tú (lista en el código) | Preprocesamiento + Naive Bayes + SVM + Random Forest + Accuracy / F1 / matriz de confusión + app web base |
| **2** | Compañero 2 | Aplicación web de análisis de texto (completa) + dashboard de sentimiento |
| **3** | Compañero 3 | Análisis temporal (gráficos por fecha) |

---

## Parte 1 — hecha

### Qué incluye

- Limpieza de texto: minúsculas, quitar URLs, menciones `@usuario`, signos y espacios de más. Los hashtags se convierten en la palabra (`#love` → `love`).
- Vectorización TF-IDF (hasta 5000 features, unigramas y bigramas).
- Tres modelos de clasificación:
  - **Naive Bayes** (`MultinomialNB`)
  - **SVM** lineal (`LinearSVC`, adecuado para texto)
  - **Random Forest**
- Métricas en conjunto de prueba: **Accuracy**, **F1**, precision, recall y **matriz de confusión**.
- App Flask con:
  - Inicio (resumen de las 3 partes)
  - Página de métricas
  - Analizar un texto (clasificación con los 3 modelos y un consenso)
  - Rutas `/dashboard` y `/temporal` como marcadores para las partes 2 y 3

### Cómo ejecutarlo

```bash
pip install -r requirements.txt
python -m src.train
python app.py
```

Abre `http://127.0.0.1:5000`.

El entrenamiento usa una **muestra balanceada de 40 000 tweets** (20 000 negativos y 20 000 positivos) para que SVM y Random Forest terminen en unos minutos. El CSV completo es enorme; no hace falta entrenar con 1.6 millones para demostrar el pipeline. Si quieren más datos, cambien `SAMPLE_SIZE` en `src/config.py`.

Al entrenar se generan:

- `data/muestra_entrenamiento.csv` — muestra con texto original, texto limpio, etiqueta y **fecha** (la Parte 3 la necesita)
- `models/tfidf_vectorizer.joblib`
- `models/naive_bayes.joblib`, `models/svm.joblib`, `models/random_forest.joblib`
- `models/metrics.json`

### Archivos de la Parte 1

```
app.py
src/config.py
src/preprocessing.py
src/dataset.py
src/train.py
src/predict.py
templates/
static/css/style.css
```

---

## Parte 2 — falta (compañero 2)

**Objetivo:** convertir la app en una herramienta real de análisis y un **dashboard de sentimiento**.

Hoy solo se puede pegar **un** texto. Dashboard y análisis masivo no existen (la ruta `/dashboard` es un placeholder).

### Qué hay que construir

1. **Análisis de varios textos**
   - Caja para pegar muchos tweets (uno por línea).
   - Subida de CSV con una columna `text` (y opcionalmente `date` para pasárselo a la Parte 3).
   - Tabla de resultados: texto, predicción de cada modelo, consenso.
   - Botón para descargar resultados en CSV.

2. **Dashboard de sentimiento**
   - Porcentaje positivo vs negativo (gráfico de barras o donut).
   - Conteo total analizado.
   - Comparación de los 3 modelos: qué tanto coinciden (por ejemplo, 3/3 de acuerdo vs empate).
   - Palabras o bigramas más frecuentes en positivos y en negativos (se puede sacar del TF-IDF o de conteos sobre `texto_limpio`).
   - Usar la misma paleta azul / fosforescente.

3. **Integración con la Parte 1 (no reentrenar)**
   - Reutilizar `src.predict.predecir_texto` y `cargar_artefactos`.
   - No duplicar el preprocesamiento: importar `limpiar_texto`.
   - Si los modelos no están entrenados, mostrar el mismo aviso (`python -m src.train`).

### Librerías sugeridas

- Gráficos en el navegador: Chart.js (CDN, sin build) o Plotly.
- Seguir con Flask y las plantillas actuales (`templates/base.html`).

### Criterio de “terminado”

- Un usuario puede subir 100 textos y ver el % de sentimiento.
- El dashboard se actualiza con esos resultados (sesión o archivo en `data/`).
- Se ve la comparación Naive Bayes vs SVM vs Random Forest.

---

## Parte 3 — falta (compañero 3)

**Objetivo:** **análisis temporal** usando la columna `date` del dataset.

Formato de fecha en Sentiment140:

```
Mon Apr 06 22:19:45 PDT 2009
```

Pandas puede parsearla, por ejemplo:

```python
df["fecha"] = pd.to_datetime(df["date"], errors="coerce")
```

`PDT` a veces molesta: si falla, quiten la zona con un replace (`" PDT"` → `""`) y vuelvan a parsear.

### Qué hay que construir

1. **Página `/temporal` de verdad**
   - Elegir granularidad: día, semana o mes.
   - Línea de tiempo: cantidad de tweets positivos y negativos.
   - Ratio de positividad por periodo (`positivos / total`).
   - Filtro de rango de fechas (el dataset es sobre todo abril–junio 2009).

2. **Fuente de datos**
   - Empezar con `data/muestra_entrenamiento.csv` (ya tiene `date`, `label`, `text`).
   - Opcional: permitir que el dashboard de la Parte 2 le pase predicciones nuevas (textos del usuario con fecha).
   - No hace falta volver a entrenar modelos. Si quieren “sentimiento predicho vs etiqueta real”, usen `src.predict` sobre la muestra.

3. **Hallazgos**
   - Destacar el día con más tweets negativos y el más positivo.
   - Un recuadro de interpretación (1–2 frases) para la presentación.

### Criterio de “terminado”

- Un gráfico de sentimiento a lo largo del tiempo que se pueda filtrar.
- Números claros (totales, picos, tendencia).
- Misma identidad visual que el resto de la app.

---

## Columnas del CSV original

El archivo no trae cabecera. Las columnas son:

| Columna | Significado |
| --- | --- |
| `target` | 0 = negativo, 4 = positivo |
| `ids` | id del tweet |
| `date` | fecha y hora |
| `flag` | query (casi siempre `NO_QUERY`) |
| `user` | usuario |
| `text` | texto del tweet |

La Parte 1 guarda `label` como 0/1 (`1` = positivo).

---

## Requisitos cubiertos vs pendientes

| Requerimiento | Estado |
| --- | --- |
| Preprocesamiento de texto | Parte 1 lista |
| Naive Bayes, SVM, Random Forest | Parte 1 lista |
| Accuracy, F1, matriz de confusión | Parte 1 lista |
| Aplicación web de análisis de texto | Base en Parte 1; masivo y UX completa = Parte 2 |
| Dashboard de sentimiento | Parte 2 |
| Análisis temporal | Parte 3 |
