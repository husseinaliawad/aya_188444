from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# ====== أساسي ======
SECRET_KEY = os.getenv("SECRET_KEY", "django-insecure-change-me-please")
DEBUG = os.getenv("DEBUG", "True") == "True"

# يمكن ضبطها من المتغيرات، والقيمة الافتراضية تشمل Render دومينك
ALLOWED_HOSTS = os.getenv(
    "ALLOWED_HOSTS",
    "localhost,127.0.0.1,aya-188444.onrender.com"
).split(",")

# Django 4+ يتطلب https في القيم
CSRF_TRUSTED_ORIGINS = os.getenv(
    "CSRF_TRUSTED_ORIGINS",
    "https://aya-188444.onrender.com,https://*.onrender.com"
).split(",")

# ====== تطبيقات ======
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # تطبيقك
    "selector",
]

# ====== Middleware (مع WhiteNoise للملفات الثابتة) ======
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # مهم لـ static على Render
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# خلف Proxy مثل Render
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True

ROOT_URLCONF = "ga_site.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "ga_site.wsgi.application"
ASGI_APPLICATION = "ga_site.asgi.application"

# ====== قاعدة البيانات ======
# استخدم SQLite افتراضيًا. (يمكنك لاحقًا تبديلها لـ Postgres عبر متغيرات البيئة)
DATABASES = {
    "default": {
        "ENGINE": os.getenv("DB_ENGINE", "django.db.backends.sqlite3"),
        "NAME": os.getenv("DB_NAME", BASE_DIR / "db.sqlite3"),
        # لإعداد Postgres مثلاً عيّن:
        # DB_ENGINE=django.db.backends.postgresql
        # DB_NAME=...
        # DB_USER=...
        # DB_PASSWORD=...
        # DB_HOST=...
        # DB_PORT=5432
        "USER": os.getenv("DB_USER", ""),
        "PASSWORD": os.getenv("DB_PASSWORD", ""),
        "HOST": os.getenv("DB_HOST", ""),
        "PORT": os.getenv("DB_PORT", ""),
    }
}

# ====== Auth ======
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ====== لغة/وقت ======
LANGUAGE_CODE = "ar"
TIME_ZONE = os.getenv("TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

# ====== Static / Media ======
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"  # مهم للنشر
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ====== أمان الكوكيز (مهم للـ CSRF خاصة لو Cross-Site) ======
if DEBUG:
    CSRF_COOKIE_SECURE = False
    SESSION_COOKIE_SECURE = False
    CSRF_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SAMESITE = "Lax"
else:
    # Production على Render عادة HTTPS
    CSRF_COOKIE_SECURE = True
    SESSION_COOKIE_SECURE = True
    # لو عندك Frontend على دومين مختلف وتحتاج إرسال الكوكي عبر المواقع:
    # عيّن القيمتين أدناه إلى "None"
    CSRF_COOKIE_SAMESITE = os.getenv("CSRF_COOKIE_SAMESITE", "None")
    SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "None")
