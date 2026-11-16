"""Ajuste de hiperparámetros por validación, usando solo el conjunto de entrenamiento.

Uso: python -m src.tuning   (requiere haber ejecutado antes src.train para tener la muestra)
Los valores ganadores se fijan a mano en src/train.py::construir_modelos.
"""
import json

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import RANDOM_STATE, RESULTS_DIR, SAMPLE_PATH
from src.train import crear_vectorizador

AJUSTE_PATH = RESULTS_DIR / "ajuste.json"


def busqueda_cv(X, y, estimador, rejilla) -> list:
    """Validación cruzada estratificada de 3 particiones; el TF-IDF se ajusta dentro
    de cada partición (Pipeline) para no filtrar vocabulario de la validación."""
    pipe = Pipeline([("tfidf", crear_vectorizador()), ("modelo", estimador)])
    cv = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
    g = GridSearchCV(pipe, {f"modelo__{k}": v for k, v in rejilla.items()}, cv=cv, scoring="accuracy", n_jobs=3)
    g.fit(X, y)
    return sorted(
        ({"params": {k.split("__")[1]: v for k, v in p.items()}, "accuracy": round(m, 4), "std": round(s, 4)}
         for p, m, s in zip(g.cv_results_["params"], g.cv_results_["mean_test_score"], g.cv_results_["std_test_score"])),
        key=lambda r: -r["accuracy"],
    )


def busqueda_rf(X, y) -> list:
    """Random Forest es costoso: una sola partición de validación (20 % de train)."""
    a_X, b_X, a_y, b_y = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    vec = crear_vectorizador()
    A, B = vec.fit_transform(a_X), vec.transform(b_X)
    resultados = []
    for k in [5_000, 20_000]:
        sel = SelectKBest(chi2, k=k).fit(A, a_y)
        for profundidad in [25, 80, None]:
            rf = RandomForestClassifier(n_estimators=100, max_depth=profundidad, n_jobs=-1, random_state=RANDOM_STATE)
            rf.fit(sel.transform(A), a_y)
            acc = accuracy_score(b_y, rf.predict(sel.transform(B)))
            resultados.append({"params": {"k": k, "max_depth": profundidad}, "accuracy": round(acc, 4)})
            print(f"  RF k={k} max_depth={profundidad}: {acc:.4f}", flush=True)
    return sorted(resultados, key=lambda r: -r["accuracy"])


def ajustar() -> dict:
    muestra = pd.read_csv(SAMPLE_PATH).fillna({"texto_limpio": ""})
    train = muestra[muestra["split"] == "train"]
    X, y = train["texto_limpio"], train["label"]
    resultado = {
        "Naive Bayes": busqueda_cv(X, y, MultinomialNB(), {"alpha": [0.1, 0.3, 0.5, 1.0]}),
        "SVM": busqueda_cv(X, y, LinearSVC(max_iter=4000, random_state=RANDOM_STATE), {"C": [0.1, 0.25, 0.5, 1.0]}),
        "Random Forest": busqueda_rf(X, y),
    }
    for nombre, filas in resultado.items():
        print(f"{nombre}: mejor {filas[0]['params']} ({filas[0]['accuracy']})")
    AJUSTE_PATH.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    return resultado


if __name__ == "__main__":
    ajustar()
