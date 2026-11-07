import json

import joblib

from src.config import LABEL_MAP, METRICS_PATH, MODEL_FILES, VECTORIZER_PATH
from src.preprocessing import limpiar_texto
from src.train import modelos_listos


def cargar_artefactos():
    if not modelos_listos():
        raise FileNotFoundError("Falta entrenar. Ejecuta: python -m src.train")
    vectorizer = joblib.load(VECTORIZER_PATH)
    modelos = {nombre: joblib.load(ruta) for nombre, ruta in MODEL_FILES.items()}
    metricas = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return vectorizer, modelos, metricas


def predecir_texto(texto: str, vectorizer, modelos) -> dict:
    limpio = limpiar_texto(texto)
    vec = vectorizer.transform([limpio])
    salidas = []
    votos = []
    for nombre, modelo in modelos.items():
        etiqueta = int(modelo.predict(vec)[0])
        votos.append(etiqueta)
        item = {
            "modelo": nombre,
            "label": etiqueta,
            "sentimiento": LABEL_MAP[etiqueta],
        }
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(vec)[0]
            item["confianza"] = round(float(max(proba)), 4)
        elif hasattr(modelo, "decision_function"):
            score = float(modelo.decision_function(vec)[0])
            item["confianza"] = round(abs(score) / (1 + abs(score)), 4)
        else:
            item["confianza"] = None
        salidas.append(item)

    consenso = 1 if sum(votos) >= 2 else 0
    return {
        "texto_original": texto,
        "texto_limpio": limpio,
        "consenso": LABEL_MAP[consenso],
        "predicciones": salidas,
    }
