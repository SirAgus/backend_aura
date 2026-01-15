# Voice Cloning API - Backend

## 1. De qué trata
Esta es una API de síntesis y clonación de voz diseñada para funcionar localmente. Utiliza el modelo **ResembleAI/chatterbox-turbo** para generar audio a partir de texto (TTS) y clonar voces mediante referencias de audio.

Soporta:
- **Clonación de voz**: Usando una muestra de audio de referencia (wav).
- **Ambiente dinámico**: Mezcla automática de sonidos de fondo (lluvia, oficina, etc.) usando AudioGen (si está disponible).
- **Control expresivo**: Ajustes de temperatura, velocidad y estilos mediante tags en el texto.
- **Gestión de voces**: Subida y almacenamiento de perfiles de voz.

## 2. Qué se necesita para iniciar
Para ejecutar este proyecto necesitas:

- **Sistema Operativo**: macOS (optimizado para Apple Silicon M1/M2/M3) o Linux.
- **Python**: Versión **3.11** (Requerida).
- **Gestor de paquetes**: **PDM** (Python Dependency Manager).
- **Dependencias del sistema**: `ffmpeg` y `pkg-config` (Necesarios para procesar audio).
- **Espacio en disco**: ~2GB libres para modelos y dependencias.
- **Variables de entorno**: Un archivo `.env` con las credenciales (ver ejemplo en el repositorio).

## 3. Comandos para iniciar

### Instalación
Si es la primera vez, instala las dependencias dentro de la carpeta `backend`:

```bash
# Instalar dependencias con PDM
pdm install
```

### Ejecución
Elige uno de los siguientes modos para levantar el servidor en `http://0.0.0.0:8000`:

```bash
# Modo Producción
pdm run start

# Modo Desarrollo (con recarga automática)
pdm run dev
```

> **Nota**: La primera vez que inicies, el servidor tardará unos minutos descargando los modelos necesarios.

## 4. Endpoints existentes y qué hacen

Todos los endpoints están protegidos por **Basic Auth** (usuario y contraseña definidos en `.env`).

### `GET /`
- **Descripción**: Verifica el estado del servidor.
- **Uso**: Health check para saber si la API está respondiendo.

### `POST /demo`
- **Descripción**: Genera un audio rápido usando una configuración por defecto.
- **Uso**: Pruebas rápidas de síntesis.
- **Parámetros**: `text`.

### `POST /generate-tts`
- **Descripción**: El endpoint principal para síntesis de voz avanzada. Permite clonar voces y configurar parámetros.
- **Uso**: Generar audio final.
- **Parámetros clave**: 
  - `text`: Texto a hablar.
  - `voice_id` o `audio_prompt`: Voz a clonar/usar.
  - `language`: Idioma del texto (opcional).
  - `temperature`, `speed`: Ajustes de la generación.
  - `ambience_id`: Id de sonido de fondo (ej: 'rain', 'office').

### `POST /voices/upload`
- **Descripción**: Sube y guarda una nueva muestra de voz clonada.
- **Uso**: Guardar voces favoritas para usarlas después por su ID.
- **Parámetros**: `name` (ID de la voz), `file` (archivo .wav).

### `GET /voices`
- **Descripción**: Lista todas las voces disponibles guardadas en el sistema.
- **Uso**: Obtener IDs de voces para usar en `/generate-tts`.

### `GET /history`
- **Descripción**: Devuelve el historial de audios generados recientemente.
- **Uso**: Revisar generaciones pasadas.

### `GET /download/{filename}`
- **Descripción**: Descarga el archivo de audio (.wav) generado.
- **Uso**: Recuperar el archivo de audio resultante de una generación.
