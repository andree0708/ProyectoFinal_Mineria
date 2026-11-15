"""Descarga el dataset Sentiment140 sin credenciales de Kaggle.

Kaggle (kazanova/sentiment140) es un espejo del archivo publicado por los
autores originales en Stanford (Go, Bhayani y Huang, 2009). Este script baja
ese mismo zip y extrae los CSV en data/raw/.

Uso:
    python -m src.download
"""
import shutil
import sys
import urllib.request
import zipfile

from src.config import DATASET_PATH, DATASET_URL, RAW_DIR

ZIP_PATH = RAW_DIR / "trainingandtestdata.zip"


def _progreso(bloques: int, tam_bloque: int, total: int) -> None:
    if total <= 0:
        return
    pct = min(100, bloques * tam_bloque * 100 // total)
    sys.stdout.write(f"\r  descargando... {pct:3d}%")
    sys.stdout.flush()


def descargar_dataset(forzar: bool = False) -> None:
    if DATASET_PATH.exists() and not forzar:
        print(f"Dataset ya presente: {DATASET_PATH}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando Sentiment140 desde {DATASET_URL}")
    parcial = ZIP_PATH.with_suffix(".part")
    urllib.request.urlretrieve(DATASET_URL, parcial, _progreso)
    print()
    shutil.move(parcial, ZIP_PATH)

    print("Extrayendo...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for miembro in zf.namelist():
            nombre = miembro.rsplit("/", 1)[-1]
            if not nombre.endswith(".csv"):
                continue
            with zf.open(miembro) as origen, open(RAW_DIR / nombre, "wb") as destino:
                shutil.copyfileobj(origen, destino)
    ZIP_PATH.unlink()

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"El zip no contenía {DATASET_PATH.name}")
    print(f"Listo: {DATASET_PATH}")


if __name__ == "__main__":
    descargar_dataset(forzar="--forzar" in sys.argv)
