import pandas as pd

from src.config import COLUMNS, DATASET_PATH, RANDOM_STATE, SAMPLE_PATH, SAMPLE_SIZE
from src.preprocessing import limpiar_texto


def cargar_dataset(ruta=None) -> pd.DataFrame:
    ruta = ruta or DATASET_PATH
    df = pd.read_csv(
        ruta,
        encoding="latin-1",
        header=None,
        names=COLUMNS,
    )
    df["label"] = (df["target"] == 4).astype(int)
    return df


def muestrear_balanceado(df: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    por_clase = max(1, n // 2)
    negativos = df[df["label"] == 0].sample(n=por_clase, random_state=RANDOM_STATE)
    positivos = df[df["label"] == 1].sample(n=por_clase, random_state=RANDOM_STATE)
    muestra = pd.concat([negativos, positivos]).sample(frac=1, random_state=RANDOM_STATE)
    return muestra.reset_index(drop=True)


def preparar_muestra() -> pd.DataFrame:
    df = cargar_dataset()
    muestra = muestrear_balanceado(df)
    muestra["texto_limpio"] = muestra["text"].map(limpiar_texto)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    muestra.to_csv(SAMPLE_PATH, index=False)
    return muestra
