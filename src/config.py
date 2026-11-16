from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Datos ---------------------------------------------------------------
# data/raw/        dataset original tal cual se descarga (no se versiona)
# data/processed/  datos limpios derivados del raw (muestra preprocesada)
# data/results/    salidas: métricas de entrenamiento, predicciones, exports
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

DATASET_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
DATASET_PATH = RAW_DIR / "training.1600000.processed.noemoticon.csv"
COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

SAMPLE_PATH = PROCESSED_DIR / "muestra_entrenamiento.csv"
METRICS_PATH = RESULTS_DIR / "metrics.json"

# --- Entrenamiento -------------------------------------------------------
SAMPLE_SIZE = 200_000  # 160 000 entrenamiento / 40 000 evaluación
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = None  # sin tope: el vocabulario se limita solo con MIN_DF
NGRAM_RANGE = (1, 2)
MIN_DF = 2
SUBLINEAR_TF = True  # 1 + log(tf): reduce el peso de términos repetidos

# --- Modelos (binarios, no se versionan) ---------------------------------
MODELS_DIR = ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
MODEL_FILES = {
    "Naive Bayes": MODELS_DIR / "naive_bayes.joblib",
    "SVM": MODELS_DIR / "svm.joblib",
    "Random Forest": MODELS_DIR / "random_forest.joblib",
}

LABEL_MAP = {0: "Negativo", 1: "Positivo"}
