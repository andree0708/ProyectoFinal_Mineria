"""Curva de aprendizaje: accuracy y coste de entrenamiento según el tamaño de la muestra.

Usa el mismo conjunto de evaluación que el resto del proyecto (split "test" de la
muestra) y conjuntos de entrenamiento balanceados y anidados tomados del resto del
dataset, sin solaparse con la evaluación. Es costoso (unos minutos), por eso no se
ejecuta en cada entrenamiento: python -m src.curva
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.config import RANDOM_STATE, RESULTS_DIR, SAMPLE_PATH, SAMPLE_SIZE, TEST_SIZE
from src.dataset import cargar_dataset
from src.preprocessing import limpiar_texto
from src.train import construir_modelos, crear_vectorizador

CURVA_PATH = RESULTS_DIR / "curva_aprendizaje.json"
TAMANOS = [10_000, 20_000, 40_000, 80_000, 160_000, 320_000, 640_000, 1_280_000]
MAX_RANDOM_FOREST = 320_000  # por encima, el bosque tarda y ocupa demasiado


def generar() -> dict:
    muestra = pd.read_csv(SAMPLE_PATH)
    test = muestra[muestra["split"] == "test"]
    X_test, y_test = test["text"].map(limpiar_texto), test["label"].to_numpy()

    df = cargar_dataset()
    df = df[~df["text"].isin(set(test["text"]))]  # nada del test entra en entrenamiento
    rng = np.random.default_rng(RANDOM_STATE)
    por_clase = {c: df[df["label"] == c].iloc[rng.permutation((df["label"] == c).sum())] for c in (0, 1)}
    tamanos = [n for n in TAMANOS if n // 2 <= min(len(v) for v in por_clase.values())]
    mayor = pd.concat([por_clase[c].iloc[: tamanos[-1] // 2] for c in (0, 1)])
    mayor = mayor.assign(limpio=mayor["text"].map(limpiar_texto))
    por_clase = {c: mayor[mayor["label"] == c] for c in (0, 1)}

    puntos = []
    for n in tamanos:
        train = pd.concat([por_clase[c].iloc[: n // 2] for c in (0, 1)])
        t0 = time.perf_counter()
        vec = crear_vectorizador()
        X = vec.fit_transform(train["limpio"])
        Xt = vec.transform(X_test)
        t_vec = time.perf_counter() - t0
        punto = {"n_train": int(n), "vocabulario": int(X.shape[1]), "modelos": {}}
        for nombre, modelo in construir_modelos().items():
            if nombre == "Random Forest" and n > MAX_RANDOM_FOREST:
                continue
            t0 = time.perf_counter()
            modelo.fit(X, train["label"])
            seg = time.perf_counter() - t0 + t_vec
            acc = accuracy_score(y_test, modelo.predict(Xt))
            punto["modelos"][nombre] = {"accuracy": round(float(acc), 4), "segundos": round(seg, 1)}
            print(f"n={n:>9,} {nombre:14} acc={acc:.4f} {seg:6.1f}s", flush=True)
        puntos.append(punto)

    resultado = {
        "n_test": int(len(test)),
        "n_train_elegido": int(SAMPLE_SIZE * (1 - TEST_SIZE)),
        "max_random_forest": MAX_RANDOM_FOREST,
        "puntos": puntos,
    }
    CURVA_PATH.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    print(f"Curva guardada en {CURVA_PATH}")
    return resultado


def resumir(curva: dict, modelo: str = "SVM") -> dict:
    """Cifras derivadas para el texto del hallazgo: ganancia media por cada duplicación
    de datos antes y después del tamaño elegido, y coste de Random Forest."""
    from math import log2

    pts = [p for p in curva["puntos"] if modelo in p["modelos"]]
    acc = lambda p: p["modelos"][modelo]["accuracy"]
    elegido = next(p for p in pts if p["n_train"] == curva["n_train_elegido"])
    primero, ultimo = pts[0], pts[-1]
    por_dup = lambda a, b: (acc(b) - acc(a)) / log2(b["n_train"] / a["n_train"])
    rf = [p for p in curva["puntos"] if "Random Forest" in p["modelos"]]
    rf_e = next((p for p in rf if p["n_train"] == elegido["n_train"]), None)
    rf_sig = next((p for p in rf if p["n_train"] == 2 * elegido["n_train"]), None)
    return {
        "modelo": modelo,
        "n_elegido": elegido["n_train"], "acc_elegido": acc(elegido),
        "n_min": primero["n_train"], "n_max": ultimo["n_train"], "acc_max": acc(ultimo),
        "factor_datos": ultimo["n_train"] // elegido["n_train"],
        "factor_tiempo": round(ultimo["modelos"][modelo]["segundos"] / elegido["modelos"][modelo]["segundos"], 1),
        "ganancia_antes": round(por_dup(primero, elegido), 4),
        "ganancia_despues": round(por_dup(elegido, ultimo), 4),
        "rf_seg_elegido": rf_e and rf_e["modelos"]["Random Forest"]["segundos"],
        "rf_seg_doble": rf_sig and rf_sig["modelos"]["Random Forest"]["segundos"],
    }


if __name__ == "__main__":
    generar()
