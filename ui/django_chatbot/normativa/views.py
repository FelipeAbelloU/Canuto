"""Vistas de la web de CANUTO: muestra las paginas y responde las preguntas."""
import json
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from src.chat.pipeline import ChatPipeline

RAIZ = Path(__file__).resolve().parent.parent.parent.parent

# El modelo se carga una vez (pesa varios GB). Las conversaciones viven en el
# navegador; el modelo recibe solo la pregunta nueva, como se entreno.
_modelo = None


def _get_modelo():
    global _modelo
    if _modelo is None:
        from src.config_loader import load_config
        from src.factory import create_model

        config = load_config(str(RAIZ / "config" / "config.yaml"))
        ruta = config.get("model", {}).get("checkpoint_path", "")
        if ruta and not Path(ruta).is_absolute():
            config["model"]["checkpoint_path"] = str(RAIZ / ruta)
        _modelo = create_model(config)
    return _modelo


def index(request):
    return render(request, "normativa/index.html")


def chat(request):
    # Dos abiertas, una con cita y una de un articulo que no existe (AS 20/2021).
    sugeridas = [
        "¿Cuál es la nota mínima para homologar una materia?",
        "¿Cuáles son los requisitos de grado?",
        "¿Qué dice el artículo 38 del Acuerdo Superior 20 de 2021?",
        "¿Qué establece el artículo 500 del Acuerdo Superior No. 007 de 1990?",
    ]
    modelo = _get_modelo()
    return render(request, "normativa/chat.html", {
        "suggested_questions": sugeridas,
        "model_version": modelo.name if modelo else "sin modelo",
    })


@csrf_exempt
@require_POST
def api_query(request):
    try:
        data = json.loads(request.body)
        pregunta = data.get("question", "").strip()
        if not pregunta:
            return JsonResponse({"error": "Pregunta vacía."}, status=400)

        # Sin historial: un pipeline nuevo por pregunta.
        respuesta = ChatPipeline(_get_modelo(), max_history=0).query(pregunta)
        return JsonResponse({"answer": respuesta["answer"]})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

