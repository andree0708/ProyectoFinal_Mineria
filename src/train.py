import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import (
    MAX_FEATURES,
    METRICS_PATH,
    RESULTS_DIR,
    SAMPLE_PATH,
    MODEL_FILES,
    MODELS_DIR,
    NGRAM_RANGE,
    RANDOM_STATE,
    TEST_SIZE,
    VECTORIZER_PATH,
)
from src.dataset import preparar_muestra


def construir_modelos() -> dict:
    return {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(C=1.0, max_iter=4000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=80,
            max_depth=25,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def metricas_modelo(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted")), 4),
        "recall": round(float(recall_score(y_true, y_pred, average="weighted")), 4),
        "matriz_confusion": cm.tolist(),
    }


def entrenar() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Cargando y preprocesando muestra del dataset...")
    muestra = preparar_muestra()

    idx_train, idx_test = train_test_split(
        muestra.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=muestra["label"],
    )
    # Se marca cada fila para que dashboard / análisis temporal puedan
    # evaluar solo sobre datos que los modelos no vieron al entrenar.
    muestra["split"] = "train"
    muestra.loc[idx_test, "split"] = "test"
    muestra.to_csv(SAMPLE_PATH, index=False)

    X_train, y_train = muestra.loc[idx_train, "texto_limpio"], muestra.loc[idx_train, "label"]
    X_test, y_test = muestra.loc[idx_test, "texto_limpio"], muestra.loc[idx_test, "label"]

    print("Vectorizando con TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words="english",
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    resultados = {
        "n_muestra": int(len(muestra)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "modelos": {},
    }

    modelos = construir_modelos()
    for nombre, modelo in modelos.items():
        print(f"Entrenando {nombre}...")
        modelo.fit(X_train_vec, y_train)
        pred = modelo.predict(X_test_vec)
        resultados["modelos"][nombre] = metricas_modelo(y_test, pred)
        joblib.dump(modelo, MODEL_FILES[nombre])
        print(f"  accuracy={resultados['modelos'][nombre]['accuracy']}  f1={resultados['modelos'][nombre]['f1']}")

    METRICS_PATH.write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    print(f"Métricas guardadas en {METRICS_PATH}")
    return resultados


def modelos_listos() -> bool:
    archivos = [VECTORIZER_PATH, METRICS_PATH, *MODEL_FILES.values()]
    return all(Path(p).exists() for p in archivos)


if __name__ == "__main__":
    entrenar()
