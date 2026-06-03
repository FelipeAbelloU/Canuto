import json
import sys
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

# Pipeline compartido entre todas las peticiones (se carga una sola vez)
_pipeline = None


def _get_pipeline():
    """Carga el pipeline de chat la primera vez que se necesita."""
    global _pipeline
    if _pipeline is None:
        config_path = Path(__file__).resolve().parent.parent.parent.parent / "config" / "config.yaml"
        from src.factory import create_pipeline
        _pipeline = create_pipeline(str(config_path))
    return _pipeline


def index(request):
    """Página principal con el widget flotante de CANUTO."""
    return render(request, "normativa/index.html")


def chat(request):
    """Chat en pantalla completa, estilo ChatGPT."""
    suggested = [
        "¿Cuáles son los requisitos de segunda lengua para graduarse?",
        "¿Cómo funciona el fraccionamiento de matrícula?",
        "¿Qué opciones de grado existen en Ciencias Básicas e Ingeniería?",
        "¿Qué debo hacer si tengo una incapacidad médica?",
        "¿Qué establece el reglamento estudiantil de pregrado?",
    ]
    return render(request, "normativa/chat.html", {"suggested_questions": suggested})


@csrf_exempt
@require_POST
def api_query(request):
    """Recibe una pregunta y devuelve la respuesta del modelo."""
    try:
        data = json.loads(request.body)
        question = data.get("question", "").strip()
        if not question:
            return JsonResponse({"error": "Pregunta vacía."}, status=400)

        pipeline = _get_pipeline()
        result = pipeline.query(question)
        return JsonResponse({"answer": result["answer"]})

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_reset(request):
    """Reinicia el historial de conversación."""
    _get_pipeline().reset_history()
    return JsonResponse({"status": "ok"})
