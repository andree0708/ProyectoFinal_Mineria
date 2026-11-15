"""Hallazgos del análisis con las métricas y pruebas estadísticas que los soportan.

Genera data/results/analisis.json, que consume la página /analisis de la app.
Replica (y cuantifica) las conclusiones del notebook 01_eda_y_evaluacion.

Uso:
    python -m src.analysis
"""
import json
import math
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2_contingency, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from src.config import COLUMNS, RAW_DIR, RESULTS_DIR, SAMPLE_PATH
from src.dataset import cargar_dataset
from src.predict import cargar_artefactos
from src.preprocessing import limpiar_texto

ANALISIS_PATH = RESULTS_DIR / "analisis.json"
MANUAL_PATH = RAW_DIR / "testdata.manual.2009.06.14.csv"
CORTE_TEMPORAL = pd.Timestamp("2009-06-17")
NEGACION_RE = re.compile(r"\b(?:not|no|never|nothing|nobody|none|cannot|\w+n't|cant|dont|wont|didnt|isnt|doesnt)\b", re.I)


def wilson(aciertos: int, n: int, z: float = 1.96) -> list:
    """Intervalo de confianza 95 % de Wilson para una proporción."""
    if n == 0:
        return [0.0, 0.0]
    p = aciertos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centro - margen, 4), round(centro + margen, 4)]


def mcnemar(y, pred_a, pred_b) -> dict:
    """Prueba exacta de McNemar: ¿dos clasificadores tienen distinta tasa de error?"""
    a_ok, b_ok = pred_a == y, pred_b == y
    b = int((a_ok & ~b_ok).sum())  # A acierta, B falla
    c = int((~a_ok & b_ok).sum())  # A falla, B acierta
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return {"a_acierta_b_falla": b, "a_falla_b_acierta": c, "p_valor": float(p)}


def tasas_error(y, pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "fpr": round(fp / (fp + tn), 4),  # negativos reales marcados como positivos
        "fnr": round(fn / (fn + tp), 4),  # positivos reales marcados como negativos
        "pct_pred_positivo": round(float(np.mean(pred)), 4),
    }


def analizar_dataset() -> dict:
    df = cargar_dataset()
    df["fecha"] = pd.to_datetime(df["date"].str.replace(" PDT", "", regex=False),
                                 format="%a %b %d %H:%M:%S %Y")
    palabras = df["text"].str.split().str.len()
    antes = df["fecha"] < CORTE_TEMPORAL
    tabla = pd.crosstab(antes, df["label"])  # filas: después/antes · columnas: neg/pos
    chi2, p, _, _ = chi2_contingency(tabla)
    cramer_v = math.sqrt(chi2 / (tabla.values.sum() * (min(tabla.shape) - 1)))

    por_dia = df.groupby(df["fecha"].dt.date)["label"].agg(["size", "mean"])
    return {
        "n_tweets": int(len(df)),
        "n_usuarios": int(df["user"].nunique()),
        "fecha_min": str(df["fecha"].min().date()),
        "fecha_max": str(df["fecha"].max().date()),
        "n_dias_con_datos": int(len(por_dia)),
        "n_dias_rango": int((df["fecha"].max() - df["fecha"].min()).days + 1),
        "pct_positivo": round(float(df["label"].mean()), 4),
        "palabras_media": {"negativo": round(float(palabras[df.label == 0].mean()), 2),
                           "positivo": round(float(palabras[df.label == 1].mean()), 2)},
        "sesgo_temporal": {
            "corte": str(CORTE_TEMPORAL.date()),
            "n_antes": int(antes.sum()),
            "n_despues": int((~antes).sum()),
            "pct_positivo_antes": round(float(df.loc[antes, "label"].mean()), 4),
            "pct_positivo_despues": round(float(df.loc[~antes, "label"].mean()), 4),
            "rango_diario_antes": [round(float(por_dia.loc[por_dia.index < CORTE_TEMPORAL.date(), "mean"].min()), 4),
                                   round(float(por_dia.loc[por_dia.index < CORTE_TEMPORAL.date(), "mean"].max()), 4)],
            "chi2": round(float(chi2), 1),
            "p_valor": float(p),
            "cramer_v": round(cramer_v, 4),
        },
        "por_dia": [{"fecha": str(f), "n": int(r["size"]), "pct_positivo": round(float(r["mean"]), 4)}
                    for f, r in por_dia.iterrows()],
    }


def analizar_modelos(vectorizer, modelos) -> dict:
    muestra = pd.read_csv(SAMPLE_PATH).fillna({"texto_limpio": ""})
    test = muestra[muestra["split"] == "test"]
    X = vectorizer.transform(test["texto_limpio"])
    y = test["label"].to_numpy()
    preds = {n: m.predict(X) for n, m in modelos.items()}
    votos = np.vstack(list(preds.values()))
    preds["Consenso"] = (votos.sum(0) >= 2).astype(int)

    por_modelo = {}
    for n, p in preds.items():
        aciertos = int((p == y).sum())
        por_modelo[n] = {
            "accuracy": round(aciertos / len(y), 4),
            "accuracy_ic95": wilson(aciertos, len(y)),
            "f1": round(float(f1_score(y, p, average="weighted")), 4),
            **tasas_error(y, p),
        }

    nombres = list(modelos)
    mejor = max(nombres, key=lambda n: por_modelo[n]["f1"])

    # Negaciones: la lista de stop words de sklearn elimina not/no/nor, y "don't" queda "don".
    con_negacion = test["text"].str.contains(NEGACION_RE).to_numpy()
    p_mejor = preds[mejor]
    acierta = p_mejor == y
    chi2_neg, p_neg, _, _ = chi2_contingency(pd.crosstab(con_negacion, acierta))
    return {
        "n_test": int(len(y)),
        "mejor_individual": mejor,
        "modelos": por_modelo,
        "comparaciones": {
            "Naive Bayes vs SVM": mcnemar(y, preds["Naive Bayes"], preds["SVM"]),
            f"{mejor} vs Random Forest": mcnemar(y, preds[mejor], preds["Random Forest"]),
            f"Consenso vs {mejor}": mcnemar(y, preds["Consenso"], preds[mejor]),
        },
        "acuerdo_3_de_3": round(float((votos == votos[0]).all(0).mean()), 4),
        "kappa": {f"{a} · {b}": round(float(cohen_kappa_score(preds[a], preds[b])), 4)
                  for i, a in enumerate(nombres) for b in nombres[i + 1:]},
        "negaciones": {
            "modelo": mejor,
            "pct_tweets_con_negacion": round(float(con_negacion.mean()), 4),
            "n_con": int(con_negacion.sum()),
            "n_sin": int((~con_negacion).sum()),
            "accuracy_con": round(float(accuracy_score(y[con_negacion], p_mejor[con_negacion])), 4),
            "accuracy_sin": round(float(accuracy_score(y[~con_negacion], p_mejor[~con_negacion])), 4),
            "chi2": round(float(chi2_neg), 2),
            "p_valor": float(p_neg),
            "stop_words_negacion_eliminadas": sorted({"not", "no", "nor", "never", "nothing", "none",
                                                      "cannot", "nobody"} & set(vectorizer.get_stop_words())),
        },
    }


def analizar_manual(vectorizer, modelos) -> dict | None:
    if not MANUAL_PATH.exists():
        return None
    m = pd.read_csv(MANUAL_PATH, encoding="latin-1", header=None, names=COLUMNS)
    n_neutros = int((m["target"] == 2).sum())
    m = m[m["target"] != 2].copy()
    m["label"] = (m["target"] == 4).astype(int)
    X = vectorizer.transform(m["text"].map(limpiar_texto))
    resultado = {"n": int(len(m)), "n_neutros_excluidos": n_neutros, "modelos": {}}
    for n, mod in modelos.items():
        p = mod.predict(X)
        m[n] = p
        aciertos = int((p == m["label"]).sum())
        resultado["modelos"][n] = {
            "accuracy": round(aciertos / len(m), 4),
            "accuracy_ic95": wilson(aciertos, len(m)),
            "f1": round(float(f1_score(m["label"], p, average="weighted")), 4),
        }

    mejor = max(resultado["modelos"], key=lambda n: resultado["modelos"][n]["f1"])
    marcas = (m.groupby("flag").agg(n=("label", "size"), real=("label", "mean"), pred=(mejor, "mean"))
              .query("n >= 10").sort_values("real", ascending=False))
    rho, p_rho = spearmanr(marcas["real"], marcas["pred"])
    resultado["marcas"] = {
        "modelo": mejor,
        "min_tweets": 10,
        "detalle": [{"marca": k, "n": int(r.n), "pct_real": round(r.real, 4), "pct_pred": round(r.pred, 4)}
                    for k, r in marcas.iterrows()],
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(p_rho),
        "mae": round(float((marcas["real"] - marcas["pred"]).abs().mean()), 4),
    }
    return resultado


def generar() -> dict:
    vectorizer, modelos, _ = cargar_artefactos()
    print("Analizando dataset completo...")
    datos = {"dataset": analizar_dataset()}
    print("Analizando modelos en el split de test...")
    datos["modelos"] = analizar_modelos(vectorizer, modelos)
    print("Evaluando en el test manual de Sentiment140...")
    datos["manual"] = analizar_manual(vectorizer, modelos)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ANALISIS_PATH.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Análisis guardado en {ANALISIS_PATH}")
    return datos


if __name__ == "__main__":
    generar()
