from django.urls import path
from . import views

urlpatterns = [
    path("",       views.index, name="index"),   # página con widget flotante
    path("chat/",  views.chat,  name="chat"),    # chat pantalla completa
    path("api/query/",  views.api_query,  name="api_query"),   # POST: pregunta → respuesta
    path("api/reset/",  views.api_reset,  name="api_reset"),   # POST: reiniciar historial
]
