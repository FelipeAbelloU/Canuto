import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Agrega la raiz de CANUTO al path para importar src.*
PROYECTO5_ROOT = BASE_DIR.parent.parent
if str(PROYECTO5_ROOT) not in sys.path:
    sys.path.insert(0, str(PROYECTO5_ROOT))

SECRET_KEY = "canuto-dev-key-cambiar-en-produccion"

DEBUG = True

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
