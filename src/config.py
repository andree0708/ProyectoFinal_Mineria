from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "training.1600000.processed.noemoticon.csv"
COLUMNS = ["target", "ids", "date", "flag", "user", "text"]

SAMPLE_SIZE = 40_000
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_FEATURES = 5_000
NGRAM_RANGE = (1, 2)

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
SAMPLE_PATH = DATA_DIR / "muestra_entrenamiento.csv"

MODEL_FILES = {
    "Naive Bayes": MODELS_DIR / "naive_bayes.joblib",
    "SVM": MODELS_DIR / "svm.joblib",
    "Random Forest": MODELS_DIR / "random_forest.joblib",
}

LABEL_MAP = {0: "Negativo", 1: "Positivo"}
