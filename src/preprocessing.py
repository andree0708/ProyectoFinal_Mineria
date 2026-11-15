import html
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
SPACES_RE = re.compile(r"\s+")

# Negaciones: se expanden las contracciones para que "don't like" y "dont like"
# queden como "do not like" y el bigrama "not like" llegue al modelo.
NEGACIONES_IRREGULARES = [
    (re.compile(r"\bcan'?t\b"), "can not"),
    (re.compile(r"\bwon'?t\b"), "will not"),
    (re.compile(r"\bain'?t\b"), "is not"),
]
NEGACION_NT_RE = re.compile(r"\b(\w+)n't\b")
NEGACION_SIN_APOSTROFE_RE = re.compile(
    r"\b(do|does|did|is|are|was|were|has|have|had|could|should|would|must|need)nt\b"
)

# Lista de stop words de scikit-learn sin las palabras de negación: sin ellas
# "not good" se reduciría a "good" y el modelo invertiría el sentido.
NEGACIONES = frozenset({"not", "no", "nor", "never", "nothing", "none", "nobody", "neither", "cannot"})
STOP_WORDS = sorted(ENGLISH_STOP_WORDS - NEGACIONES)


def expandir_negaciones(texto: str) -> str:
    texto = texto.replace("’", "'")
    for patron, reemplazo in NEGACIONES_IRREGULARES:
        texto = patron.sub(reemplazo, texto)
    texto = NEGACION_NT_RE.sub(r"\1 not", texto)
    return NEGACION_SIN_APOSTROFE_RE.sub(r"\1 not", texto)


def limpiar_texto(texto: str, negaciones: bool = True) -> str:
    """Limpia un tweet: minúsculas, negaciones, URLs, menciones, hashtags y signos.

    `negaciones=False` reproduce la limpieza original (sin expandir contracciones);
    solo se usa en src.analysis para medir el efecto de la corrección.
    """
    if texto is None:
        return ""
    texto = html.unescape(str(texto)).lower()  # &quot; &amp; &lt; → " & <
    texto = URL_RE.sub(" ", texto)
    texto = MENTION_RE.sub(" ", texto)
    if negaciones:
        texto = expandir_negaciones(texto)
    texto = HASHTAG_RE.sub(r"\1", texto)
    texto = NON_ALNUM_RE.sub(" ", texto)
    texto = SPACES_RE.sub(" ", texto).strip()
    return texto
