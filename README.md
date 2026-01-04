# Voice Cloning API - Backend

API de síntesis y clonación de voz usando **ResembleAI/chatterbox-turbo**.

## 🗣️ Soporte de Idiomas

### Configuración por Variable de Entorno

### Modelo TTS Disponible

Solo está disponible **ChatterboxTurboTTS** (versión 0.1.6 del paquete):

- ✅ **Funciona perfectamente** con voces de referencia
- ✅ **Español soportado** vía voces chilenas (`agus`, `agus_latin`)
- ❌ **Modelo multilingüe NO disponible** en esta versión del paquete

### Nota Importante

El modelo multilingüe (`ChatterboxMultilingualTTS`) **no está incluido** en `chatterbox-tts 0.1.6`. Solo está disponible el modelo Turbo, que funciona correctamente con voces de referencia para español.

### Modelos Disponibles

#### **Turbo (Predeterminado)**
- **Español**: ✅ Funciona vía voces de referencia
- **Voces**: Chilenas (`agus`, `agus_latin`)
- **Ventaja**: Siempre disponible y funcional

#### **Multilingual (Avanzado)**
- **Español**: ✅ Soporte nativo con parámetro `language`
- **Idiomas**: 23+ idiomas soportados
- **Ventaja**: Mejor pronunciación y prosodia
- **Nota**: Requiere instalación especial

### Mejorar Acento Español

Para mejor acento, sube voces más naturales:

```bash
# Subir voz española más natural
curl -X POST "http://localhost:8000/voices/upload" \
  -u admin:upfint2001 \
  -F "name=voz_espana" \
  -F "language=es" \
  -F "region=ES" \
  -F "file=@voz_mas_natural.wav"
```

## 📋 Requisitos

- **Python 3.11** (recomendado)
- **macOS** con Apple Silicon (ARM64) o sistema compatible
- Al menos **2GB de espacio libre** para modelos y dependencias

## 🚀 Instalación Rápida

### Opción 1: Script automático (Recomendado)

```bash
cd backend
chmod +x setup_and_run.sh
./setup_and_run.sh
```

Este script hace todo automáticamente:
- ✅ Activa el entorno virtual
- ✅ Instala todas las dependencias
- ✅ Verifica la instalación
- ✅ Inicia el servidor

### Opción 2: Instalación manual

#### 1. Crear entorno virtual

```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate
```

#### 2. Instalar dependencias

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Nota**: La instalación puede tardar varios minutos debido a las dependencias de PyTorch y el modelo de TTS.

### 3. Configurar variables de entorno

El archivo `.env` ya está creado con valores por defecto. Puedes editarlo para cambiar las credenciales:

```bash
API_USERNAME=admin
API_PASSWORD=upfint2001
API_KEY=your_secret_api_key_here
STORAGE_DIR=outputs
VOICES_DIR=voices
HISTORY_FILE=history.json
TTS_MODEL=turbo  # Solo turbo disponible
```

## ▶️ Iniciar el servidor

```bash
source venv/bin/activate
python main.py
```

El servidor se iniciará en `http://0.0.0.0:8000`

### Primera ejecución

Al iniciar por primera vez, el servidor:
1. Descargará automáticamente 3 voces de ejemplo:
   - `female_english` - Voz femenina en inglés
   - `male_english` - Voz masculina en inglés
   - `female_spanish` - Voz femenina en español
2. Creará las carpetas `outputs/` y `voices/`
3. Cargará el modelo **Chatterbox-Turbo** (puede tardar 1-2 minutos)

## 📡 Endpoints disponibles

### 🔐 Autenticación

Todos los endpoints requieren **HTTP Basic Auth** con las credenciales del archivo `.env`.

---

### `GET /`

Verifica el estado del servidor.

**Respuesta:**
```json
{
  "status": "Voice Backend Running",
  "model": "ResembleAI/chatterbox-turbo",
  "auth": "enabled"
}
```

---

### `POST /demo`

Genera audio usando la voz femenina por defecto.

**Parámetros:**
- `text` (Form): Texto a sintetizar

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/demo" \
  -u admin:admin_password \
  -F "text=Hola, esta es una prueba de voz" \
  --output demo.wav
```

---

### `POST /generate-tts`

Genera audio con una voz personalizada.

**Parámetros:**
- `text` (Form): Texto a sintetizar
- `audio_prompt` (File, opcional): Archivo de audio para clonar la voz
- `voice_id` (Form, opcional): ID de una voz guardada previamente

**Ejemplo con voz guardada:**
```bash
curl -X POST "http://localhost:8000/generate-tts" \
  -u admin:admin_password \
  -F "text=Hello world" \
  -F "voice_id=female_english" \
  --output output.wav
```

**Ejemplo con archivo de audio:**
```bash
curl -X POST "http://localhost:8000/generate-tts" \
  -u admin:admin_password \
  -F "text=Hello world" \
  -F "audio_prompt=@mi_voz.wav" \
  --output output.wav
```

---

### `POST /voices/upload`

Guarda una muestra de voz para reutilizarla.

**Parámetros:**
- `name` (Form): Nombre identificador de la voz
- `file` (File): Archivo de audio (.wav)

**Ejemplo:**
```bash
curl -X POST "http://localhost:8000/voices/upload" \
  -u admin:admin_password \
  -F "name=mi_voz" \
  -F "file=@sample.wav"
```

**Respuesta:**
```json
{
  "status": "Voice clone saved locally",
  "voice_id": "mi_voz"
}
```

---

### `GET /voices`

Lista todas las voces guardadas.

**Ejemplo:**
```bash
curl -X GET "http://localhost:8000/voices" \
  -u admin:admin_password
```

**Respuesta:**
```json
{
  "voices": ["female_english", "male_english", "female_spanish", "mi_voz"]
}
```

---

### `GET /history`

Obtiene el historial de generaciones.

**Ejemplo:**
```bash
curl -X GET "http://localhost:8000/history" \
  -u admin:admin_password
```

**Respuesta:**
```json
[
  {
    "id": "abc123",
    "text": "Hello world",
    "filename": "gen_abc123.wav",
    "voice_used": "female_english",
    "timestamp": "2026-01-04T13:00:00",
    "user": "admin"
  }
]
```

---

### `GET /download/{filename}`

Descarga un archivo de audio del historial.

**Ejemplo:**
```bash
curl -X GET "http://localhost:8000/download/gen_abc123.wav" \
  -u admin:admin_password \
  --output archivo.wav
```

---

## 📁 Estructura de archivos

```
backend/
├── main.py              # Servidor FastAPI
├── requirements.txt     # Dependencias
├── .env                 # Configuración (credenciales)
├── .gitignore          
├── venv/               # Entorno virtual
├── outputs/            # Audios generados
├── voices/             # Muestras de voz guardadas
└── history.json        # Registro de operaciones
```

## 🛠️ Solución de problemas

### Error: `python: command not found`
Usa `python3.11` en lugar de `python`.

### Error al instalar dependencias
Asegúrate de tener Python 3.11. Python 3.14 no es compatible con algunas dependencias.

### El modelo tarda mucho en cargar
Es normal. La primera vez que se carga el modelo puede tardar 1-2 minutos.

### No se descargan las voces por defecto
Verifica tu conexión a internet. Las voces se descargan desde GitHub.

## 📝 Notas

- Todos los archivos se almacenan **localmente** en las carpetas del proyecto
- El modelo requiere un **audio de referencia** para clonar voces (no genera voces desde cero)
- Los audios de referencia deben ser archivos `.wav` de al menos 3-5 segundos
- El servidor usa **CPU por defecto** (CUDA si está disponible)

## 🔒 Seguridad

- Cambia las credenciales en `.env` antes de usar en producción
- El archivo `.env` está en `.gitignore` para evitar exponer credenciales
- Todos los endpoints requieren autenticación

## 📚 Documentación adicional

- [Chatterbox-Turbo en Hugging Face](https://huggingface.co/ResembleAI/chatterbox-turbo)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
