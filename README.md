# Sentinel Blue — Análisis de sentimiento en tweets

Proyecto final en equipo (3 personas) sobre el dataset **Sentiment140** (`training.1600000.processed.noemoticon.csv`): 1.6 millones de tweets etiquetados como negativo (`target = 0`) o positivo (`target = 4`).

Estilo visual **editorial claro**: fondo crema `#f7f3ec`, tinta `#1b1815`, positivo `#2a7f6f`, negativo `#d9573b`.
Tipografías: **Fraunces** (títulos), **Inter** (texto), **JetBrains Mono** (números). Todo está en
`static/css/style.css` como variables CSS; los gráficos usan **Apache ECharts** con el tema de `static/js/charts.js`.

Las fuentes y ECharts están **alojados en el propio proyecto** (`static/fonts`, `static/vendor`). No se carga nada
de Google Fonts ni de CDNs, así que no hay rastreo de terceros y la app funciona sin internet (útil para la presentación).

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
  - Página de métricas (con fórmulas e IC 95 %)
  - Página de análisis: hallazgos con su evidencia (McNemar, χ², kappa, Spearman, IC de Wilson)
  - Analizar un texto (clasificación con los 3 modelos y un consenso)
  - Rutas `/dashboard` y `/temporal` como marcadores para las partes 2 y 3

### Cómo ejecutarlo

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
python -m src.download   # baja Sentiment140 (~80 MB) a data/raw/ — sin cuenta de Kaggle
python -m src.train      # si falta el dataset, lo descarga solo
python app.py
```

Para el notebook de análisis (gráficos de EDA, matrices de confusión, percepción por marca):

```bash
pip install -r requirements-notebook.txt
jupyter lab notebooks/01_eda_y_evaluacion.ipynb
```

El dataset de Kaggle es un espejo del zip original de Stanford; `src/download.py` baja ese zip directamente, así que no hacen falta credenciales.

### Estructura de carpetas

```
data/
  raw/         dataset original descargado (no se sube a git)
  processed/   muestra_entrenamiento.csv: texto limpio + etiqueta + fecha + split (train/test)
  results/     metrics.json y salidas de la app (predicciones, exports)
models/        modelos .joblib entrenados (no se suben a git)
src/analysis.py   hallazgos + pruebas estadísticas → data/results/analisis.json (página /analisis)
notebooks/     01_eda_y_evaluacion.ipynb: análisis exploratorio + evaluación (para informe/presentación)
src/           pipeline: config, download, preprocessing, dataset, train, predict
templates/, static/   frontend Flask
```

Abre `http://127.0.0.1:5000`.

El entrenamiento usa una **muestra balanceada de 40 000 tweets** (20 000 negativos y 20 000 positivos) para que SVM y Random Forest terminen en unos minutos. El CSV completo es enorme; no hace falta entrenar con 1.6 millones para demostrar el pipeline. Si quieren más datos, cambien `SAMPLE_SIZE` en `src/config.py`.

Al entrenar se generan:

- `data/processed/muestra_entrenamiento.csv` — muestra con texto original, texto limpio, etiqueta, **fecha** (la Parte 3 la necesita) y `split` (`train`/`test`: usar `test` para evaluar sin fuga de datos)
- `models/tfidf_vectorizer.joblib`
- `models/naive_bayes.joblib`, `models/svm.joblib`, `models/random_forest.joblib`
- `data/results/metrics.json`

### Archivos de la Parte 1

```
app.py
src/config.py
src/download.py
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
- El dashboard se actualiza con esos resultados (sesión o archivo en `data/results/`).
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
   - Empezar con `data/processed/muestra_entrenamiento.csv` (ya tiene `date`, `label`, `text`).
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
