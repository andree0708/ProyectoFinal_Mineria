from flask import Flask, render_template, request

from src.predict import cargar_artefactos, predecir_texto
from src.train import modelos_listos

app = Flask(__name__)

_cache = {"vectorizer": None, "modelos": None, "metricas": None}


def obtener_artefactos():
    if not modelos_listos():
        return None, None, None
    if _cache["vectorizer"] is None:
        _cache["vectorizer"], _cache["modelos"], _cache["metricas"] = cargar_artefactos()
    return _cache["vectorizer"], _cache["modelos"], _cache["metricas"]


@app.route("/")
def inicio():
    return render_template("index.html", listos=modelos_listos())


@app.route("/metricas")
def metricas():
    _, _, datos = obtener_artefactos()
    return render_template(
        "metricas.html",
        listos=modelos_listos(),
        datos=datos,
    )


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
    return render_template(
        "analizar.html",
        listos=modelos_listos(),
        resultado=resultado,
        error=error,
        texto=texto,
    )


@app.route("/dashboard")
def dashboard():
    return render_template("pendiente.html", listos=modelos_listos(), parte=2)


@app.route("/temporal")
def temporal():
    return render_template("pendiente.html", listos=modelos_listos(), parte=3)


if __name__ == "__main__":
    app.run(debug=True)
