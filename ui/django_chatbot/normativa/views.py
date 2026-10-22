"""Vistas de la web de CANUTO: muestra las paginas y responde las preguntas."""
import json
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from src.chat.pipeline import SIN_MODELO

RAIZ = Path(__file__).resolve().parent.parent.parent.parent

# El modelo se carga una sola vez porque pesa varios GB. Las conversaciones las
# guarda el navegador (chat.html); el modelo recibe solo la pregunta nueva,
# porque se entreno con una pregunta a la vez.
_modelo = None


def _get_modelo():
    """Carga el modelo la primera vez que se necesita."""
    global _modelo
    if _modelo is None:
        from src.config_loader import load_config
        from src.factory import create_model

        config = load_config(str(RAIZ / "config" / "config.yaml"))
        # La ruta del checkpoint viene relativa a la raiz del proyecto.
        ruta = config.get("model", {}).get("checkpoint_path", "")
        if ruta and not Path(ruta).is_absolute():
            config["model"]["checkpoint_path"] = str(RAIZ / ruta)
        _modelo = create_model(config)
    return _modelo


def index(request):
    """Pagina de inicio con el widget flotante."""
    return render(request, "normativa/index.html")


def chat(request):
    """Chat en pantalla completa."""
    # Preguntas de ejemplo de la bienvenida, del Reglamento Estudiantil (AS 20 de
    # 2021): dos abiertas (arts. 46 y 61) y una con cita (art. 38), para mostrar
    # las dos formas de preguntar. La ultima es de un articulo que no existe,
    # para mostrar que reconoce lo que no sabe.
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
    """Recibe la pregunta y devuelve la respuesta del modelo."""
    try:
        data = json.loads(request.body)
        pregunta = data.get("question", "").strip()
        if not pregunta:
            return JsonResponse({"error": "Pregunta vacía."}, status=400)

        modelo = _get_modelo()
        if modelo is None or not modelo.is_available():
            return JsonResponse({"answer": SIN_MODELO})
        respuesta = modelo.chat([{"role": "user", "content": pregunta}])
        return JsonResponse({"answer": respuesta})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

