import re

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
SPACES_RE = re.compile(r"\s+")


def limpiar_texto(texto: str) -> str:
    """Limpia un tweet: minúsculas, URLs, menciones, hashtags y signos."""
    if texto is None:
        return ""
    texto = str(texto).lower()
    texto = URL_RE.sub(" ", texto)
    texto = MENTION_RE.sub(" ", texto)
    texto = HASHTAG_RE.sub(r"\1", texto)
    texto = NON_ALNUM_RE.sub(" ", texto)
    texto = SPACES_RE.sub(" ", texto).strip()
    return texto
