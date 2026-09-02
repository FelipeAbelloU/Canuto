"""Vistas de la web de CANUTO: muestra las paginas y responde las preguntas."""
import json
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

RAIZ = Path(__file__).resolve().parent.parent.parent.parent

# El modelo se carga una sola vez porque pesa varios GB, pero cada usuario tiene
# su propia conversacion para que no se le mezcle el historial con el de otro.
_modelo = None
_max_history = 5
_conversaciones = {}


def _get_modelo():
    """Carga el modelo la primera vez que se necesita."""
    global _modelo, _max_history
    if _modelo is None:
        from src.config_loader import load_config
        from src.factory import create_model

        config = load_config(str(RAIZ / "config" / "config.yaml"))
        # La ruta del checkpoint viene relativa a la raiz del proyecto.
        ruta = config.get("model", {}).get("checkpoint_path", "")
        if ruta and not Path(ruta).is_absolute():
            config["model"]["checkpoint_path"] = str(RAIZ / ruta)
        _max_history = config.get("chat", {}).get("max_history", 5)
        _modelo = create_model(config)
    return _modelo


def _get_chat(usuario):
    """Devuelve el chat de ese usuario y lo crea si es su primera pregunta."""
    if usuario not in _conversaciones:
        from src.chat.pipeline import ChatPipeline
        modelo = _get_modelo()
        _conversaciones[usuario] = ChatPipeline(modelo, _max_history)
    return _conversaciones[usuario]


def index(request):
    """Pagina de inicio con el widget flotante."""
    return render(request, "normativa/index.html")


def chat(request):
    """Chat en pantalla completa."""
    sugeridas = [  # preguntas de ejemplo que aparecen en la bienvenida
        "¿Cuáles son los requisitos de segunda lengua para graduarse?",
        "¿Cómo funciona el fraccionamiento de matrícula?",
        "¿Qué opciones de grado existen en Ciencias Básicas e Ingeniería?",
        "¿Qué debo hacer si tengo una incapacidad médica?",
        "¿Qué establece el reglamento estudiantil de pregrado?",
    ]
    modelo = _get_modelo()
    return render(request, "normativa/chat.html", {
        "suggested_questions": sugeridas,
        "model_version": modelo.name if modelo else "sin modelo",
    })


@csrf_exempt
@require_POST
def api_query(request):
    """Recibe la pregunta y devuelve la respuesta del modelo."""
    try:
        data = json.loads(request.body)
        pregunta = data.get("question", "").strip()
        if not pregunta:
            return JsonResponse({"error": "Pregunta vacía."}, status=400)

        resultado = _get_chat(data.get("user_id", "anonimo")).query(pregunta)
        return JsonResponse({"answer": resultado["answer"]})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@csrf_exempt
@require_POST
def api_reset(request):
    """Borra el historial de ese usuario para empezar una conversacion nueva."""
    data = json.loads(request.body or "{}")
    _get_chat(data.get("user_id", "anonimo")).reset_history()
    return JsonResponse({"status": "ok"})
