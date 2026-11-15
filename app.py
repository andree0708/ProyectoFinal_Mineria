import json
from datetime import date, timedelta

from flask import Flask, abort, render_template, request

from src.analysis import ANALISIS_PATH
from src.dashboard import FUENTES, resumir_lote
from src.predict import cargar_artefactos, predecir_texto
from src.train import modelos_listos

app = Flask(__name__)

_cache = {"vectorizer": None, "modelos": None, "metricas": None, "dashboard": {}}

MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def obtener_artefactos():
    if not modelos_listos():
        return None, None, None
    if _cache["vectorizer"] is None:
        _cache["vectorizer"], _cache["modelos"], _cache["metricas"] = cargar_artefactos()
    return _cache["vectorizer"], _cache["modelos"], _cache["metricas"]


def cargar_analisis():
    if not ANALISIS_PATH.exists():
        return None
    return json.loads(ANALISIS_PATH.read_text(encoding="utf-8"))


# --- Filtros de formato para las plantillas --------------------------------
@app.template_filter("pct")
def fmt_pct(valor, decimales=1):
    return f"{valor * 100:.{decimales}f}\u00a0%"


@app.template_filter("miles")
def fmt_miles(valor):
    return f"{valor:,}".replace(",", " ")


@app.template_filter("fecha")
def fmt_fecha(iso, desplazamiento_dias=0, con_anio=True):
    """'2009-06-17' -> '17 de junio de 2009' (desplazamiento opcional en días)."""
    f = date.fromisoformat(iso) + timedelta(days=desplazamiento_dias)
    texto = f"{f.day} de {MESES[f.month - 1]}"
    return f"{texto} de {f.year}" if con_anio else texto


@app.template_filter("pval")
def fmt_pval(p):
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


@app.context_processor
def contexto_global():
    return {"listos": modelos_listos(), "ruta": request.endpoint}


# --- Rutas -----------------------------------------------------------------
@app.route("/")
def inicio():
    _, _, datos = obtener_artefactos()
    return render_template("index.html", datos=datos, analisis=cargar_analisis())


@app.route("/metricas")
def metricas():
    _, _, datos = obtener_artefactos()
    return render_template("metricas.html", datos=datos, analisis=cargar_analisis())


@app.route("/analisis")
def analisis():
    return render_template("analisis.html", a=cargar_analisis())


@app.route("/analizar", methods=["GET", "POST"])
def analizar():
    resultado = None
    error = None
    texto = ""
    if request.method == "POST":
        texto = (request.form.get("texto") or "").strip()
        if not modelos_listos():
            error = "Primero entrena los modelos con: python -m src.train"
        elif not texto:
            error = "Escribe un texto para analizar."
        else:
            vectorizer, modelos, _ = obtener_artefactos()
            resultado = predecir_texto(texto, vectorizer, modelos)
    return render_template("analizar.html", resultado=resultado, error=error, texto=texto)


@app.route("/dashboard")
def dashboard():
    fuente = request.args.get("fuente", "evaluacion")
    if fuente not in FUENTES:
        abort(404)
    resumen = None
    if modelos_listos():
        if fuente not in _cache["dashboard"]:
            datos = FUENTES[fuente]["cargar"]()
            if datos is not None:
                textos, etiquetas, grupos = datos
                vectorizer, modelos, _ = obtener_artefactos()
                _cache["dashboard"][fuente] = resumir_lote(textos, vectorizer, modelos, etiquetas, grupos)
        resumen = _cache["dashboard"].get(fuente)
    return render_template("dashboard.html", r=resumen, fuente=fuente, fuentes=FUENTES)


@app.route("/temporal")
def temporal():
    return render_template("pendiente.html", parte=3)


if __name__ == "__main__":
    app.run(debug=True)
