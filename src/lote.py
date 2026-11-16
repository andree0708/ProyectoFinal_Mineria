"""Lectura de datos aportados por el usuario (texto pegado o CSV) para el dashboard.

El CSV no necesita un formato fijo. Las columnas se reconocen por nombre:
  - texto (obligatoria): text, texto, tweet, mensaje, comentario, review...
  - sentimiento (opcional): label, sentimiento, sentiment, target, polaridad...
  - grupo (opcional): marca, brand, tema, topic, producto, categoria...
  - fecha (opcional): date, fecha, created_at... se conserva en la descarga.
Si ninguna columna se llama como un texto, se usa la de contenido más largo.
También acepta el formato original de Sentiment140 (6 columnas sin cabecera).
"""
import csv
import html
import io

import numpy as np
import pandas as pd

from src.config import COLUMNS, SAMPLES_DIR

MAX_FILAS = 50_000
MAX_BYTES = 10 * 1024 * 1024

NOMBRES_TEXTO = {"text", "texto", "tweet", "tweets", "mensaje", "message", "content", "contenido",
                 "comentario", "comment", "review", "resena", "reseña", "opinion", "opinión", "post"}
NOMBRES_ETIQUETA = {"label", "etiqueta", "sentimiento", "sentiment", "target", "polaridad", "polarity",
                    "clase", "class", "y"}
NOMBRES_GRUPO = {"marca", "brand", "grupo", "group", "tema", "topic", "query", "producto", "product",
                 "categoria", "categoría", "category", "empresa", "company"}

NOMBRES_FECHA = {"date", "fecha", "created_at", "timestamp", "datetime", "time"}

POSITIVOS = {"positivo", "positive", "pos", "p", "+", "bueno", "good"}
NEGATIVOS = {"negativo", "negative", "neg", "n", "-", "malo", "bad"}


# Ejemplos disponibles en samples/ (el nombre del archivo es la clave que envía el formulario).
EJEMPLOS = {
    "marcas_etiquetadas.csv": {
        "titulo": "Marcas etiquetadas",
        "descripcion": "Tweets sobre marcas y temas, con el sentimiento asignado por personas.",
        "incluye": ["Sentimiento", "Grupo"],
    },
    "sentiment140_original.csv": {
        "titulo": "Formato original de Sentiment140",
        "descripcion": "Archivo sin cabecera, con seis columnas y sentimiento 0/2/4. Los neutrales se excluyen de las métricas.",
        "incluye": ["Sin cabecera", "Sentimiento", "Grupo", "Fecha", "Neutrales"],
    },
    "tweets_sin_etiqueta.csv": {
        "titulo": "Tweets sin etiquetar",
        "descripcion": "Solo texto y fecha. El panel muestra estimaciones, sin métricas de precisión.",
        "incluye": ["Fecha"],
    },
    "resenas_productos.csv": {
        "titulo": "Reseñas de productos",
        "descripcion": "Reseñas ficticias con columnas en español, separadas por punto y coma.",
        "incluye": ["Separador ;", "Sentimiento", "Grupo", "Fecha", "Neutrales"],
    },
}


def listar_ejemplos(lineas_vista: int = 5, max_caracteres: int = 110) -> list[dict]:
    """Ejemplos presentes en disco, con número de filas y las primeras líneas tal cual."""
    lista = []
    for archivo, meta in EJEMPLOS.items():
        ruta = SAMPLES_DIR / archivo
        if not ruta.exists():
            continue
        lineas = _decodificar(ruta.read_bytes()).splitlines()
        sin_cabecera = "Sin cabecera" in meta["incluye"]
        vista = [l if len(l) <= max_caracteres else l[: max_caracteres - 1] + "…" for l in lineas[:lineas_vista]]
        lista.append({**meta, "archivo": archivo, "filas": len(lineas) - (0 if sin_cabecera else 1), "vista": vista})
    return lista


def leer_ejemplo(archivo: str) -> bytes:
    if archivo not in EJEMPLOS or not (SAMPLES_DIR / archivo).exists():
        raise ErrorDatos("El ejemplo seleccionado no está disponible.")
    return (SAMPLES_DIR / archivo).read_bytes()


class ErrorDatos(ValueError):
    """Error con un mensaje apto para mostrar al usuario."""


def _decodificar(contenido: bytes) -> str:
    for codificacion in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return contenido.decode(codificacion)
        except UnicodeDecodeError:
            continue
    raise ErrorDatos("No se pudo leer el archivo. Guárdalo como CSV con codificación UTF-8.")


def _buscar(columnas, nombres):
    for c in columnas:
        if str(c).strip().lower() in nombres:
            return c
    return None


def normalizar_etiquetas(serie: pd.Series) -> pd.Series:
    """Devuelve 1 (positivo), 0 (negativo) o NaN (neutral o no reconocido)."""
    numerica = pd.to_numeric(serie, errors="coerce")
    if numerica.notna().mean() > 0.9:
        valores = set(numerica.dropna().unique())
        if valores <= {0, 2, 4}:  # convención de Sentiment140
            return numerica.map({0: 0, 4: 1})
        if valores <= {-1, 0, 1} and -1 in valores:
            return numerica.map({-1: 0, 1: 1})
        if valores <= {0, 1}:
            return numerica.astype(float)
        return pd.Series(np.nan, index=serie.index)
    texto = serie.astype(str).str.strip().str.lower()
    return texto.map(lambda v: 1 if v in POSITIVOS else (0 if v in NEGATIVOS else np.nan))


def _formato_sentiment140(df: pd.DataFrame) -> bool:
    if df.shape[1] != len(COLUMNS):
        return False
    primera = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    return primera.notna().all() and set(primera.unique()) <= {0, 2, 4}


def leer_csv(contenido: bytes) -> tuple[pd.DataFrame, dict]:
    if len(contenido) > MAX_BYTES:
        raise ErrorDatos(f"El archivo supera el máximo de {MAX_BYTES // 2**20} MB.")
    texto = _decodificar(contenido)
    if not texto.strip():
        raise ErrorDatos("El archivo está vacío.")
    try:
        separador = csv.Sniffer().sniff(texto[:8192], delimiters=",;\t|").delimiter
    except csv.Error:
        separador = ","

    try:
        sin_cabecera = pd.read_csv(io.StringIO(texto), sep=separador, header=None, dtype=str,
                                   keep_default_na=False, on_bad_lines="skip")
    except pd.errors.ParserError as e:
        raise ErrorDatos("No se pudo interpretar el archivo como CSV.") from e
    if _formato_sentiment140(sin_cabecera):
        df = sin_cabecera.set_axis(COLUMNS, axis=1)
        grupos = df["flag"].str.strip('"')
        return pd.DataFrame({
            "texto": df["text"],
            "etiqueta": normalizar_etiquetas(df["target"]),
            "grupo": grupos.where(grupos != "NO_QUERY"),
            "fecha": df["date"],
        }), {"formato": "Sentiment140", "texto": "text", "etiqueta": "target", "grupo": "flag", "fecha": "date"}

    df = pd.read_csv(io.StringIO(texto), sep=separador, dtype=str, keep_default_na=False, on_bad_lines="skip")
    col_texto = _buscar(df.columns, NOMBRES_TEXTO)
    if col_texto is None:
        if df.shape[1] == 1:
            # Una sola columna sin nombre reconocible: probablemente no tiene cabecera.
            df = sin_cabecera.set_axis(["primera columna"], axis=1)
            col_texto = "primera columna"
        else:
            col_texto = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
    col_etiqueta = _buscar([c for c in df.columns if c != col_texto], NOMBRES_ETIQUETA)
    col_grupo = _buscar([c for c in df.columns if c not in (col_texto, col_etiqueta)], NOMBRES_GRUPO)
    col_fecha = _buscar([c for c in df.columns if c not in (col_texto, col_etiqueta, col_grupo)], NOMBRES_FECHA)

    datos = pd.DataFrame({"texto": df[col_texto]})
    if col_etiqueta is not None:
        datos["etiqueta"] = normalizar_etiquetas(df[col_etiqueta])
    if col_grupo is not None:
        datos["grupo"] = df[col_grupo].astype(str).str.strip().replace("", np.nan)
    if col_fecha is not None:
        datos["fecha"] = df[col_fecha]
    nombre = lambda c: None if c is None else str(c)
    return datos, {"formato": "CSV", "texto": str(col_texto), "etiqueta": nombre(col_etiqueta),
                   "grupo": nombre(col_grupo), "fecha": nombre(col_fecha)}


def leer_texto_pegado(texto: str) -> tuple[pd.DataFrame, dict]:
    lineas = [l.strip() for l in texto.splitlines()]
    return pd.DataFrame({"texto": [l for l in lineas if l]}), {"formato": "Texto pegado", "texto": None,
                                                               "etiqueta": None, "grupo": None, "fecha": None}


def preparar(datos: pd.DataFrame, info: dict) -> tuple[pd.DataFrame, dict]:
    """Limpia filas vacías, aplica el límite de filas y descarta columnas sin datos útiles."""
    datos = datos.copy()
    datos["texto"] = datos["texto"].astype(str).map(html.unescape).str.strip()  # &amp; -> &
    datos = datos[datos["texto"] != ""].reset_index(drop=True)
    if datos.empty:
        raise ErrorDatos("No se encontró ningún texto para analizar.")
    info = {**info, "n_original": len(datos), "recortado": len(datos) > MAX_FILAS}
    datos = datos.head(MAX_FILAS)
    if "etiqueta" in datos and datos["etiqueta"].notna().sum() == 0:
        info["etiqueta_descartada"] = info["etiqueta"]
        datos, info["etiqueta"] = datos.drop(columns="etiqueta"), None
    if "grupo" in datos and datos["grupo"].nunique() < 2:
        datos, info["grupo"] = datos.drop(columns="grupo"), None
    if "etiqueta" in datos:
        info["n_etiquetados"] = int(datos["etiqueta"].notna().sum())
    return datos, info
