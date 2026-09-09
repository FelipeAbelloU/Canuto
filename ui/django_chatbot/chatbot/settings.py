import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Agrega la raiz de CANUTO al path para importar src.*
CANUTO_ROOT = BASE_DIR.parent.parent
if str(CANUTO_ROOT) not in sys.path:
    sys.path.insert(0, str(CANUTO_ROOT))

SECRET_KEY = "canuto-dev-key-cambiar-en-produccion"

# Servidor de desarrollo: con DEBUG en True runserver sirve las imagenes y el CSS.
# Si algun dia se publica en internet hay que poner False y servir los estaticos aparte.
DEBUG = True

# Abierto para poder abrir la pagina desde otro equipo de la red del laboratorio.
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "normativa",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "chatbot.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
            ],
        },
    },
]

WSGI_APPLICATION = "chatbot.wsgi.application"

STATIC_URL = "/static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
